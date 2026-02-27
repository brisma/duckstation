// SPDX-FileCopyrightText: 2019-2026 Connor McLaughlin <stenzek@gmail.com> and contributors.
// SPDX-License-Identifier: CC-BY-NC-ND-4.0

#include "mcp_server.h"
#include "bus.h"
#include "cdrom.h"
#include "cpu_core.h"
#include "cpu_core_private.h"
#include "cpu_disasm.h"
#include "dma.h"
#include "gpu.h"
#include "host.h"
#include "settings.h"
#include "spu.h"
#include "system.h"
#include "timers.h"

#include "achievements.h"
#include "cheats.h"
#include "controller.h"
#include "game_database.h"
#include "gte.h"
#include "interrupt_controller.h"
#include "mdec.h"
#include "memory_card.h"
#include "memory_card_image.h"
#include "pad.h"
#include "timing_event.h"

#include "util/cd_image.h"
#include "util/image.h"
#include "util/ini_settings_interface.h"
#include "util/input_manager.h"
#include "util/iso_reader.h"
#include "util/media_capture.h"
#include "util/postprocessing.h"

#include "bios.h"
#include "core.h"
#include "game_list.h"
#include "memory_scanner.h"

#include "common/assert.h"
#include "common/file_system.h"
#include "common/path.h"
#include "common/log.h"
#include "common/small_string.h"
#include "common/string_util.h"

#include "util/sockets.h"

#include "common/json_reader.h"
#include "common/json_writer.h"

#include "fmt/format.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <functional>
#include <optional>
#include <random>
#include <string>
#include <vector>

LOG_CHANNEL(MCPServer);

// NOTE: The socket multiplexer is polled from the core thread (in System::InternalExecution),
// so all OnRead() callbacks and tool handlers execute on the core thread. No additional
// dispatch is needed for direct CPU/memory/hardware state access.

struct ToolResult
{
  std::string text;
  int error_code = 0;

  bool IsError() const { return error_code != 0; }

  static ToolResult Error(int code, std::string_view message)
  {
    return ToolResult{std::string(message), code};
  }
};

namespace MCPServer {

/// Maximum HTTP body size: 4MB.
static constexpr size_t MAX_HTTP_BODY_SIZE = 4 * 1024 * 1024;

/// Maximum number of concurrent SSE clients.
static constexpr size_t MAX_SSE_CLIENTS = 4;

/// Frontend callback for post-tool-call UI refresh.
static StateChangedCallback s_state_changed_callback;

/// Optional Bearer token for authentication. Empty = no auth required.
static std::string s_auth_token;

/// Allowed CORS origin. Empty string uses default "*" for backward compatibility.
static std::string s_cors_origin;

/// Active MCP session ID (crypto-secure, assigned on initialize).
static std::string s_mcp_session_id;

/// Negotiated protocol version for the current session.
static std::string s_negotiated_protocol_version;

void SetStateChangedCallback(StateChangedCallback callback)
{
  s_state_changed_callback = std::move(callback);
}

namespace {

class ClientSocket final : public BufferedStreamSocket
{
public:
  ClientSocket(SocketMultiplexer& multiplexer, SocketDescriptor descriptor);
  ~ClientSocket() override;

  void OnSystemPaused();
  void OnSystemResumed();

  bool IsSSE() const { return m_is_sse; }
  void SendSSEEvent(std::string_view event_name, std::string_view data_json);

protected:
  void OnConnected() override;
  void OnDisconnected(const Error& error) override;
  void OnRead() override;

private:
  void ProcessHttpRequest(const std::string& method, const std::string& path,
                          const std::string& body);
  void ProcessJsonRpc(const std::string& json_body);

  void SendHttpResponse(int status_code, std::string_view content_type, std::string_view body);
  void SendHttpResponseDirect(int status_code, std::string_view content_type, std::string_view body);
  void SendHttpResponseWithSessionId(int status_code, std::string_view content_type, std::string_view body);

  static std::string MakeJsonRpcResponse(const JsonValue& id, std::string_view result_json);
  static std::string MakeJsonRpcError(const JsonValue& id, int code, std::string_view message);

  static std::string GenerateSessionId();

  std::string m_recv_buffer;
  bool m_is_sse = false;
  bool m_last_auth_ok = false;
  bool m_respond_via_sse = false;   // True during Streamable HTTP SSE request processing

  // Per-request parsed headers (set in OnRead, consumed in ProcessHttpRequest).
  bool m_accept_sse = false;
  std::string m_origin_header;
  std::string m_mcp_session_id_header;
  std::string m_mcp_protocol_version_header;
};

} // namespace

static std::shared_ptr<ListenSocket> s_mcp_listen_socket;
static std::vector<std::shared_ptr<ClientSocket>> s_mcp_clients;
static std::vector<std::shared_ptr<ClientSocket>> s_sse_clients;

// ---- VRAM write watchpoints ----

struct VRAMWatch
{
  u32 id;
  u16 x, y, width, height;
};

static std::vector<VRAMWatch> s_vram_watches;
static u32 s_next_vram_watch_id = 1;
static bool s_vram_watch_hit_pending = false;
static u16 s_vram_watch_hit_x = 0, s_vram_watch_hit_y = 0;
static u16 s_vram_watch_hit_w = 0, s_vram_watch_hit_h = 0;
static u32 s_vram_watch_hit_pc = 0;
static u32 s_vram_watch_hit_ra = 0;
static u32 s_vram_watch_hit_sp = 0;
static std::array<u32, 32> s_vram_watch_hit_regs = {};
static std::array<u32, 64> s_vram_watch_hit_stack = {}; // 256 bytes of stack from SP

// ---- Controller input state ----

struct PendingAutoRelease
{
  u32 slot;
  u32 bind_index;
  u32 frames_remaining;
};

static std::vector<PendingAutoRelease> s_auto_releases;

struct InputSequenceStep
{
  std::vector<u32> bind_indices;
  u32 duration_frames;
};

struct ActiveInputSequence
{
  u32 id;
  u32 slot;
  std::vector<InputSequenceStep> steps;
  u32 current_step;
  u32 frames_in_step;
};

static std::optional<ActiveInputSequence> s_active_sequence;
static u32 s_next_sequence_id = 1;

// ---- Log streaming state ----

static Log::Level s_log_stream_level = Log::Level::None;
static bool s_log_stream_active = false;

// ---- Resource subscription state ----

static std::vector<std::string> s_subscribed_resources;

// ---- Memory scanner state ----

static MemoryScan s_memory_scan;
static MemoryWatchList s_memory_watch_list;

// ---- Memory snapshot state ----

static std::vector<u8> s_memory_snapshot;
static u32 s_snapshot_base_address = 0;
static u32 s_snapshot_size = 0;

// ---- Temp file tracking ----

static std::vector<std::string> s_mcp_temp_files;

// ---- Utility helpers for tool handlers ----

static std::string GetMCPTempDir()
{
  return Path::Combine(EmuFolders::Cache, "mcp");
}

static std::string GetMCPTempFilePath(const char* prefix, const char* extension)
{
  const std::string dir = GetMCPTempDir();
  if (!FileSystem::DirectoryExists(dir.c_str()))
    FileSystem::EnsureDirectoryExists(dir.c_str(), false, nullptr);

  std::string path =
    Path::Combine(dir, fmt::format("{}_{:X}.{}", prefix, Timer::GetCurrentValue(), extension));
  s_mcp_temp_files.push_back(path);
  return path;
}

static void CleanupMCPTempFiles()
{
  for (const std::string& path : s_mcp_temp_files)
    FileSystem::DeleteFile(path.c_str());
  s_mcp_temp_files.clear();

  // Remove the mcp/ subdirectory itself (only succeeds if empty).
  const std::string dir = GetMCPTempDir();
  if (FileSystem::DirectoryExists(dir.c_str()))
    FileSystem::DeleteDirectory(dir.c_str());
}

static std::optional<u32> ParseAddress(const JsonValue& val)
{
  if (val.is_number_unsigned())
    return static_cast<u32>(val.get_uint());
  if (val.is_number_integer())
    return static_cast<u32>(val.get_int());
  if (val.is_string())
  {
    const std::string s = std::string(val.get_string());
    if (s.size() > 2 && (s[0] == '0' && (s[1] == 'x' || s[1] == 'X')))
      return StringUtil::FromChars<u32>(std::string_view(s).substr(2), 16);
    return StringUtil::FromChars<u32>(s);
  }
  return std::nullopt;
}

static std::string FormatHex32(u32 val)
{
  return fmt::format("0x{:08X}", val);
}

// ---- Controller helpers ----

static Controller* GetControllerForSlot(u32 slot)
{
  return Pad::GetController(slot);
}

static std::optional<u32> ResolveBindIndex(Controller* controller, ControllerType type,
                                           const std::string& button_name)
{
  const Controller::ControllerInfo& info = Controller::GetControllerInfo(type);
  for (const auto& binding : info.bindings)
  {
    if (StringUtil::EqualNoCase(binding.name, button_name))
      return binding.bind_index;
  }
  return std::nullopt;
}

// ---- Breakpoint type parsing helper ----

static std::optional<CPU::BreakpointType> ParseBreakpointType(const JsonValue& args)
{
  std::string type_str = "execute";
  if (args.contains("type") && args["type"].is_string())
    type_str = std::string(args["type"].get_string());

  if (type_str == "execute")
    return CPU::BreakpointType::Execute;
  if (type_str == "read")
    return CPU::BreakpointType::Read;
  if (type_str == "write")
    return CPU::BreakpointType::Write;
  return std::nullopt;
}

// ---- Controller Tool Handlers ----
static ToolResult HandlePressButton(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (!args.contains("button") || !args["button"].is_string())
    return ToolResult::Error(-32602, "Missing 'button' parameter");

  u32 slot = 0;
  if (args.contains("slot") && args["slot"].is_number_unsigned())
    slot = static_cast<u32>(args["slot"].get_uint());

  if (slot >= NUM_CONTROLLER_AND_CARD_PORTS)
    return ToolResult::Error(-32602, "Invalid slot number");

  Controller* controller = GetControllerForSlot(slot);
  if (!controller)
    return ToolResult::Error(-2, fmt::format("No controller in slot {}", slot));

  const std::string button_name = std::string(args["button"].get_string());
  const ControllerType type = controller->GetType();
  const std::optional<u32> bind_index = ResolveBindIndex(controller, type, button_name);
  if (!bind_index.has_value())
    return ToolResult::Error(-32602, fmt::format("Unknown button '{}'", button_name));

  controller->SetBindState(bind_index.value(), 1.0f);

  if (args.contains("duration_frames") && args["duration_frames"].is_number_unsigned())
  {
    u32 duration = static_cast<u32>(args["duration_frames"].get_uint());
    if (duration > 0)
      s_auto_releases.push_back({slot, bind_index.value(), duration});
  }

  JsonWriter w;
  w.StartObject();
  w.KeyUint("slot", slot);
  w.KeyString("button", button_name);
  w.KeyString("state", "pressed");
  if (args.contains("duration_frames") && args["duration_frames"].is_number_unsigned())
    w.KeyUint("auto_release_frames", static_cast<u32>(args["duration_frames"].get_uint()));
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleReleaseButton(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (!args.contains("button") || !args["button"].is_string())
    return ToolResult::Error(-32602, "Missing 'button' parameter");

  u32 slot = 0;
  if (args.contains("slot") && args["slot"].is_number_unsigned())
    slot = static_cast<u32>(args["slot"].get_uint());

  if (slot >= NUM_CONTROLLER_AND_CARD_PORTS)
    return ToolResult::Error(-32602, "Invalid slot number");

  Controller* controller = GetControllerForSlot(slot);
  if (!controller)
    return ToolResult::Error(-2, fmt::format("No controller in slot {}", slot));

  const std::string button_name = std::string(args["button"].get_string());
  const ControllerType type = controller->GetType();
  const std::optional<u32> bind_index = ResolveBindIndex(controller, type, button_name);
  if (!bind_index.has_value())
    return ToolResult::Error(-32602, fmt::format("Unknown button '{}'", button_name));

  controller->SetBindState(bind_index.value(), 0.0f);

  JsonWriter w;
  w.StartObject();
  w.KeyUint("slot", slot);
  w.KeyString("button", button_name);
  w.KeyString("state", "released");
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleSetAnalog(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (!args.contains("stick") || !args["stick"].is_string())
    return ToolResult::Error(-32602, "Missing 'stick' parameter (\"left\" or \"right\")");
  if (!args.contains("x") || !args["x"].is_number())
    return ToolResult::Error(-32602, "Missing 'x' parameter (float -1.0 to 1.0)");
  if (!args.contains("y") || !args["y"].is_number())
    return ToolResult::Error(-32602, "Missing 'y' parameter (float -1.0 to 1.0)");

  u32 slot = 0;
  if (args.contains("slot") && args["slot"].is_number_unsigned())
    slot = static_cast<u32>(args["slot"].get_uint());

  if (slot >= NUM_CONTROLLER_AND_CARD_PORTS)
    return ToolResult::Error(-32602, "Invalid slot number");

  Controller* controller = GetControllerForSlot(slot);
  if (!controller)
    return ToolResult::Error(-2, fmt::format("No controller in slot {}", slot));

  const std::string stick = std::string(args["stick"].get_string());
  const float x = std::clamp(static_cast<float>(args["x"].get_float()), -1.0f, 1.0f);
  const float y = std::clamp(static_cast<float>(args["y"].get_float()), -1.0f, 1.0f);

  const ControllerType type = controller->GetType();

  // Map stick name to half-axis binding names.
  // Left stick: LLeft, LRight, LDown, LUp
  // Right stick: RLeft, RRight, RDown, RUp
  const char* left_name;
  const char* right_name;
  const char* down_name;
  const char* up_name;

  if (StringUtil::EqualNoCase(stick, "left"))
  {
    left_name = "LLeft";
    right_name = "LRight";
    down_name = "LDown";
    up_name = "LUp";
  }
  else if (StringUtil::EqualNoCase(stick, "right"))
  {
    left_name = "RLeft";
    right_name = "RRight";
    down_name = "RDown";
    up_name = "RUp";
  }
  else
  {
    return ToolResult::Error(-32602, "Invalid stick name, use \"left\" or \"right\"");
  }

  // Convert x/y to half-axis values.
  const float left_val = (x < 0.0f) ? -x : 0.0f;
  const float right_val = (x > 0.0f) ? x : 0.0f;
  const float up_val = (y < 0.0f) ? -y : 0.0f;
  const float down_val = (y > 0.0f) ? y : 0.0f;

  auto set_half_axis = [&](const char* name, float value) {
    const std::optional<u32> idx = ResolveBindIndex(controller, type, name);
    if (idx.has_value())
      controller->SetBindState(idx.value(), value);
  };

  set_half_axis(left_name, left_val);
  set_half_axis(right_name, right_val);
  set_half_axis(up_name, up_val);
  set_half_axis(down_name, down_val);

  JsonWriter w;
  w.StartObject();
  w.KeyUint("slot", slot);
  w.KeyString("stick", stick);
  w.KeyDouble("x", x);
  w.KeyDouble("y", y);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleGetControllerState(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  u32 slot = 0;
  if (args.contains("slot") && args["slot"].is_number_unsigned())
    slot = static_cast<u32>(args["slot"].get_uint());

  if (slot >= NUM_CONTROLLER_AND_CARD_PORTS)
    return ToolResult::Error(-32602, "Invalid slot number");

  Controller* controller = GetControllerForSlot(slot);
  if (!controller)
    return ToolResult::Error(-2, fmt::format("No controller in slot {}", slot));

  const ControllerType type = controller->GetType();
  const Controller::ControllerInfo& info = Controller::GetControllerInfo(type);

  const u32 button_bits = controller->GetButtonStateBits();
  const std::optional<u32> analog_bytes = controller->GetAnalogInputBytes();

  JsonWriter w;
  w.StartObject();
  w.KeyUint("slot", slot);
  w.KeyString("type", info.name);

  w.Key("buttons");
  w.StartObject();
  for (const auto& binding : info.bindings)
  {
    if (binding.type == InputBindingInfo::Type::Button)
      w.KeyBool(binding.name, ((button_bits & (1u << binding.bind_index)) != 0));
  }
  w.EndObject();

  if (analog_bytes.has_value())
  {
    const u32 ab = analog_bytes.value();
    w.Key("analog");
    w.StartObject();
    w.KeyUint("byte0", ab & 0xFF);
    w.KeyUint("byte1", (ab >> 8) & 0xFF);
    w.KeyUint("byte2", (ab >> 16) & 0xFF);
    w.KeyUint("byte3", (ab >> 24) & 0xFF);
    w.EndObject();
  }

  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleListControllers([[maybe_unused]] const JsonValue& args)
{
  JsonWriter w;
  w.StartObject();
  w.Key("controllers");
  w.StartArray();

  for (u32 i = 0; i < NUM_CONTROLLER_AND_CARD_PORTS; i++)
  {
    const ControllerType type = g_settings.controller_types[i];
    if (type == ControllerType::None)
      continue;

    const Controller::ControllerInfo& info = Controller::GetControllerInfo(type);

    w.StartObject();
    w.KeyUint("slot", i);
    w.KeyString("type", info.name);
    w.KeyString("display_name", info.display_name);

    w.Key("bindings");
    w.StartArray();
    for (const auto& binding : info.bindings)
    {
      w.StartObject();
      w.KeyString("name", binding.name);
      w.KeyString("display_name", binding.display_name);
      w.KeyUint("bind_index", binding.bind_index);
      switch (binding.type)
      {
        case InputBindingInfo::Type::Button:
          w.KeyString("type", "button");
          break;
        case InputBindingInfo::Type::Axis:
          w.KeyString("type", "axis");
          break;
        case InputBindingInfo::Type::HalfAxis:
          w.KeyString("type", "half_axis");
          break;
        case InputBindingInfo::Type::Motor:
          w.KeyString("type", "motor");
          break;
        default:
          w.KeyString("type", "other");
          break;
      }
      w.EndObject();
    }
    w.EndArray();

    w.EndObject();
  }

  w.EndArray();
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleInputSequence(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (!args.contains("sequence") || !args["sequence"].is_array())
    return ToolResult::Error(-32602, "Missing 'sequence' array parameter");

  if (s_active_sequence.has_value())
    return ToolResult::Error(-1, "An input sequence is already active");

  u32 slot = 0;
  if (args.contains("slot") && args["slot"].is_number_unsigned())
    slot = static_cast<u32>(args["slot"].get_uint());

  if (slot >= NUM_CONTROLLER_AND_CARD_PORTS)
    return ToolResult::Error(-32602, "Invalid slot number");

  Controller* controller = GetControllerForSlot(slot);
  if (!controller)
    return ToolResult::Error(-2, fmt::format("No controller in slot {}", slot));

  const ControllerType type = controller->GetType();
  const JsonValue& seq_array = args["sequence"];

  std::vector<InputSequenceStep> steps;
  u32 total_frames = 0;

  for (const auto& step_json : seq_array)
  {
    if (!step_json.contains("buttons") || !step_json["buttons"].is_array())
      return ToolResult::Error(-32602, "Each step must have a 'buttons' array");
    if (!step_json.contains("duration_frames") || !step_json["duration_frames"].is_number_unsigned())
      return ToolResult::Error(-32602, "Each step must have 'duration_frames'");

    InputSequenceStep step;
    step.duration_frames = static_cast<u32>(step_json["duration_frames"].get_uint());
    if (step.duration_frames == 0)
      return ToolResult::Error(-32602, "duration_frames must be > 0");

    for (const auto& btn : step_json["buttons"])
    {
      if (!btn.is_string())
        return ToolResult::Error(-32602, "Button names must be strings");

      const std::string btn_name = std::string(btn.get_string());
      const std::optional<u32> idx = ResolveBindIndex(controller, type, btn_name);
      if (!idx.has_value())
        return ToolResult::Error(-32602, fmt::format("Unknown button '{}'", btn_name));

      step.bind_indices.push_back(idx.value());
    }

    total_frames += step.duration_frames;
    steps.push_back(std::move(step));
  }

  if (steps.empty())
    return ToolResult::Error(-32602, "Sequence must have at least one step");

  const u32 seq_id = s_next_sequence_id++;

  // Activate the first step immediately.
  for (u32 idx : steps[0].bind_indices)
    controller->SetBindState(idx, 1.0f);

  s_active_sequence = ActiveInputSequence{seq_id, slot, std::move(steps), 0, 0};

  JsonWriter w;
  w.StartObject();
  w.KeyUint("sequence_id", seq_id);
  w.KeyUint("total_frames", total_frames);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

// ---- Settings Tool Handlers ----

static ToolResult HandleGetSettings([[maybe_unused]] const JsonValue& args)
{
  JsonWriter w;
  w.StartObject();

  w.Key("GPU");
  w.StartObject();
  w.KeyString("renderer", Settings::GetRendererName(g_settings.gpu_renderer));
  w.KeyUint("resolution_scale", g_settings.gpu_resolution_scale);
  w.KeyString("texture_filter", Settings::GetTextureFilterName(g_settings.gpu_texture_filter));
  w.KeyUint("multisamples", g_settings.gpu_multisamples);
  w.KeyBool("pgxp_enable", static_cast<bool>(g_settings.gpu_pgxp_enable));
  w.KeyBool("pgxp_depth_buffer", static_cast<bool>(g_settings.gpu_pgxp_depth_buffer));
  w.KeyBool("widescreen_rendering", static_cast<bool>(g_settings.gpu_widescreen_rendering));
  w.KeyBool("widescreen_hack", static_cast<bool>(g_settings.gpu_widescreen_hack));
  w.EndObject();

  w.Key("Display");
  w.StartObject();
  w.KeyString("aspect_ratio", std::string(Settings::GetDisplayAspectRatioName(g_settings.display_aspect_ratio)));
  w.KeyString("crop_mode", Settings::GetDisplayCropModeName(g_settings.display_crop_mode));
  w.KeyBool("vsync", static_cast<bool>(g_settings.display_vsync));
  w.EndObject();

  w.Key("CPU");
  w.StartObject();
  w.KeyString("execution_mode", Settings::GetCPUExecutionModeName(g_settings.cpu_execution_mode));
  w.KeyBool("overclock_enable", static_cast<bool>(g_settings.cpu_overclock_enable));
  w.KeyUint("overclock_numerator", g_settings.cpu_overclock_numerator);
  w.KeyUint("overclock_denominator", g_settings.cpu_overclock_denominator);
  w.EndObject();

  w.Key("Emulation");
  w.StartObject();
  w.KeyDouble("emulation_speed", g_settings.emulation_speed);
  w.KeyDouble("fast_forward_speed", g_settings.fast_forward_speed);
  w.KeyDouble("turbo_speed", g_settings.turbo_speed);
  w.KeyBool("fast_boot", static_cast<bool>(g_settings.bios_patch_fast_boot));
  w.KeyString("region", Settings::GetConsoleRegionName(g_settings.region));
  w.EndObject();

  w.Key("Audio");
  w.StartObject();
  w.KeyUint("output_volume", g_settings.audio_output_volume);
  w.KeyBool("muted", static_cast<bool>(g_settings.audio_output_muted));
  w.EndObject();

  w.Key("CDROM");
  w.StartObject();
  w.KeyUint("read_speedup", g_settings.cdrom_read_speedup);
  w.KeyUint("seek_speedup", g_settings.cdrom_seek_speedup);
  w.KeyBool("mute_cd_audio", static_cast<bool>(g_settings.cdrom_mute_cd_audio));
  w.EndObject();

  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleSetSetting(const JsonValue& args)
{
  if (!args.contains("section") || !args["section"].is_string())
    return ToolResult::Error(-32602, "Missing 'section' parameter");
  if (!args.contains("key") || !args["key"].is_string())
    return ToolResult::Error(-32602, "Missing 'key' parameter");
  if (!args.contains("value"))
    return ToolResult::Error(-32602, "Missing 'value' parameter");

  const std::string section = std::string(args["section"].get_string());
  const std::string key = std::string(args["key"].get_string());
  const std::string setting_path = fmt::format("{}/{}", section, key);

  bool applied = false;

  if (setting_path == "GPU/ResolutionScale")
  {
    if (!args["value"].is_number_unsigned())
      return ToolResult::Error(-32602, "ResolutionScale must be an unsigned integer");
    const u32 scale = args["value"].get_uint();
    if (scale < 1 || scale > 16)
      return ToolResult::Error(-32602, "GPU/ResolutionScale must be between 1 and 16");
    g_settings.gpu_resolution_scale = static_cast<u8>(scale);
    applied = true;
  }
  else if (setting_path == "GPU/PGXP")
  {
    if (!args["value"].is_bool())
      return ToolResult::Error(-32602, "PGXP must be a boolean");
    g_settings.gpu_pgxp_enable = args["value"].get_bool();
    applied = true;
  }
  else if (setting_path == "CPU/Overclock")
  {
    if (!args["value"].is_bool())
      return ToolResult::Error(-32602, "Overclock must be a boolean");
    g_settings.cpu_overclock_enable = args["value"].get_bool();
    System::UpdateOverclock();
    applied = true;
  }
  else if (setting_path == "CPU/OverclockNumerator")
  {
    if (!args["value"].is_number_unsigned())
      return ToolResult::Error(-32602, "OverclockNumerator must be an unsigned integer");
    g_settings.cpu_overclock_numerator = static_cast<u32>(args["value"].get_uint());
    System::UpdateOverclock();
    applied = true;
  }
  else if (setting_path == "CPU/OverclockDenominator")
  {
    if (!args["value"].is_number_unsigned())
      return ToolResult::Error(-32602, "OverclockDenominator must be an unsigned integer");
    g_settings.cpu_overclock_denominator = static_cast<u32>(args["value"].get_uint());
    System::UpdateOverclock();
    applied = true;
  }
  else if (setting_path == "Audio/OutputVolume")
  {
    if (!args["value"].is_number_unsigned())
      return ToolResult::Error(-32602, "OutputVolume must be an unsigned integer (0-100)");
    g_settings.audio_output_volume = static_cast<u8>(std::min(static_cast<u32>(args["value"].get_uint()), 100u));
    applied = true;
  }
  else if (setting_path == "Audio/Muted")
  {
    if (!args["value"].is_bool())
      return ToolResult::Error(-32602, "Muted must be a boolean");
    g_settings.audio_output_muted = args["value"].get_bool();
    applied = true;
  }
  else if (setting_path == "CDROM/ReadSpeedup")
  {
    if (!args["value"].is_number_unsigned())
      return ToolResult::Error(-32602, "ReadSpeedup must be an unsigned integer");
    g_settings.cdrom_read_speedup = static_cast<u8>(args["value"].get_uint());
    applied = true;
  }
  else if (setting_path == "CDROM/SeekSpeedup")
  {
    if (!args["value"].is_number_unsigned())
      return ToolResult::Error(-32602, "SeekSpeedup must be an unsigned integer");
    g_settings.cdrom_seek_speedup = static_cast<u8>(args["value"].get_uint());
    applied = true;
  }
  else
  {
    return ToolResult::Error(-32602,
                             fmt::format("Unsupported setting: {}. Supported: GPU/ResolutionScale, GPU/PGXP, "
                                         "CPU/Overclock, CPU/OverclockNumerator, CPU/OverclockDenominator, "
                                         "Audio/OutputVolume, Audio/Muted, CDROM/ReadSpeedup, CDROM/SeekSpeedup",
                                         setting_path));
  }

  if (applied)
    System::ApplySettings(false);

  JsonWriter w;
  w.StartObject();
  w.KeyString("setting", setting_path);
  w.KeyBool("applied", applied);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleSetSpeed(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (args.contains("speed") && args["speed"].is_number())
  {
    const float speed = static_cast<float>(args["speed"].get_float());
    if (speed <= 0.0f || speed > 100.0f)
      return ToolResult::Error(-32602, "Speed must be between 0.01 and 100.0");
    g_settings.emulation_speed = speed;
    System::UpdateSpeedLimiterState();
  }

  if (args.contains("fast_forward") && args["fast_forward"].is_bool())
    System::SetFastForwardEnabled(args["fast_forward"].get_bool());

  if (args.contains("turbo") && args["turbo"].is_bool())
    System::SetTurboEnabled(args["turbo"].get_bool());

  if (args.contains("rewind") && args["rewind"].is_bool())
    System::SetRewindState(args["rewind"].get_bool());

  JsonWriter w;
  w.StartObject();
  w.KeyDouble("target_speed", System::GetTargetSpeed());
  w.KeyBool("fast_forward", System::IsFastForwardEnabled());
  w.KeyBool("turbo", System::IsTurboEnabled());
  w.KeyBool("rewind", System::IsRewinding());
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleTakeScreenshot(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  const char* path = nullptr;
  std::string path_str;
  if (args.contains("path") && args["path"].is_string())
  {
    path_str = std::string(args["path"].get_string());
    path = path_str.c_str();
  }

  System::SaveScreenshot(path);

  JsonWriter w;
  w.StartObject();
  w.KeyString("status", "screenshot_saved");
  if (path)
    w.KeyString("path", path_str);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

// ---- Cheat Tool Handlers ----

static ToolResult HandleListCheats(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  bool want_cheats = true;
  bool want_patches = true;
  if (args.contains("type") && args["type"].is_string())
  {
    const std::string type = std::string(args["type"].get_string());
    if (type == "cheats")
      want_patches = false;
    else if (type == "patches")
      want_cheats = false;
    // "all" keeps both true
  }

  const std::string serial = System::GetGameSerial();
  const GameHash hash = System::GetGameHash();

  JsonWriter w;
  w.StartObject();
  w.Key("codes");
  w.StartArray();

  u32 count = 0;

  if (want_cheats)
  {
    const Cheats::CodeInfoList cheat_list = Cheats::GetCodeInfoList(serial, hash, true, true, true);
    for (const auto& code : cheat_list)
    {
      w.StartObject();
      w.KeyString("name", code.name);
      w.KeyString("author", code.author);
      w.KeyString("description", code.description);
      w.KeyString("type", "cheat");
      w.KeyString("activation", (code.activation == Cheats::CodeActivation::Manual) ? "manual" : "end_frame");
      w.KeyBool("from_database", code.from_database);
      w.EndObject();
      count++;
    }
  }

  if (want_patches)
  {
    const Cheats::CodeInfoList patch_list = Cheats::GetCodeInfoList(serial, hash, false, true, true);
    for (const auto& code : patch_list)
    {
      w.StartObject();
      w.KeyString("name", code.name);
      w.KeyString("author", code.author);
      w.KeyString("description", code.description);
      w.KeyString("type", "patch");
      w.KeyString("activation", (code.activation == Cheats::CodeActivation::Manual) ? "manual" : "end_frame");
      w.KeyBool("from_database", code.from_database);
      w.EndObject();
      count++;
    }
  }

  w.EndArray();
  w.KeyUint("count", count);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleApplyCheat(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (!args.contains("name") || !args["name"].is_string())
    return ToolResult::Error(-32602, "Missing 'name' parameter");

  const std::string name = std::string(args["name"].get_string());

  if (!Cheats::ApplyManualCode(name))
    return ToolResult::Error(-2, fmt::format("Failed to apply cheat: {}", name));

  JsonWriter w;
  w.StartObject();
  w.KeyString("status", "applied");
  w.KeyString("name", name);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleToggleCheat(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (!args.contains("name") || !args["name"].is_string())
    return ToolResult::Error(-32602, "Missing 'name' parameter");

  const std::string name = std::string(args["name"].get_string());

  // Optional: explicit enable/disable, or auto-toggle if not specified.
  std::optional<bool> want_enabled;
  if (args.contains("enabled") && args["enabled"].is_bool())
    want_enabled = args["enabled"].get_bool();

  auto lock = Core::GetSettingsLock();
  SettingsInterface* sif = Core::GetGameSettingsLayer();
  if (!sif)
    return ToolResult::Error(-2, "No game settings available (no game loaded?)");

  // Check current state.
  const std::vector<std::string> enabled_list = sif->GetStringList("Cheats", "Enable");
  const bool currently_enabled =
    (std::find(enabled_list.begin(), enabled_list.end(), name) != enabled_list.end());
  const bool new_enabled = want_enabled.value_or(!currently_enabled);

  if (new_enabled != currently_enabled)
  {
    if (new_enabled)
      sif->AddToStringList("Cheats", "Enable", name.c_str());
    else
      sif->RemoveFromStringList("Cheats", "Enable", name.c_str());

    static_cast<INISettingsInterface*>(sif)->Save();
  }

  lock.unlock();

  // Reload cheats to apply the change.
  Cheats::ReloadCheats(true, true, true, true, false);

  JsonWriter w;
  w.StartObject();
  w.KeyString("name", name);
  w.KeyBool("enabled", new_enabled);
  w.KeyBool("cheats_master_enabled", Cheats::AreCheatsEnabled());
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleGetCheatStatus([[maybe_unused]] const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  JsonWriter w;
  w.StartObject();
  w.KeyBool("cheats_enabled", Cheats::AreCheatsEnabled());
  w.KeyUint("active_cheat_count", Cheats::GetActiveCheatCount());
  w.KeyUint("active_patch_count", Cheats::GetActivePatchCount());
  w.KeyBool("widescreen_patch_active", Cheats::IsWidescreenPatchActive());
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

// ---- Media Tool Handlers ----

static ToolResult HandleInsertDisc(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (!args.contains("path") || !args["path"].is_string())
    return ToolResult::Error(-32602, "Missing 'path' parameter");

  const std::string path = std::string(args["path"].get_string());

  if (!FileSystem::FileExists(path.c_str()))
    return ToolResult::Error(-2, fmt::format("File not found: {}", path));

  if (!System::InsertMedia(path.c_str()))
    return ToolResult::Error(-2, fmt::format("Failed to insert disc: {}", path));

  JsonWriter w;
  w.StartObject();
  w.KeyString("status", "inserted");
  w.KeyString("path", path);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleEjectDisc([[maybe_unused]] const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  System::RemoveMedia();

  JsonWriter w;
  w.StartObject();
  w.KeyString("status", "ejected");
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleSwitchDisc(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (args.contains("index") && args["index"].is_number_unsigned())
  {
    const u32 index = static_cast<u32>(args["index"].get_uint());
    if (!System::SwitchMediaSubImage(index))
      return ToolResult::Error(-2, fmt::format("Failed to switch to disc index {}", index));

    JsonWriter w;
    w.StartObject();
    w.KeyString("status", "switched");
    w.KeyUint("index", index);
    w.EndObject();
    return ToolResult{w.TakeOutput()};
  }
  else if (args.contains("direction") && args["direction"].is_string())
  {
    const std::string direction = std::string(args["direction"].get_string());
    bool success = false;

    if (direction == "next")
      success = System::SwitchToNextDisc(false);
    else if (direction == "previous")
      success = System::SwitchToPreviousDisc(false);
    else
      return ToolResult::Error(-32602, "Invalid direction. Use 'next' or 'previous'.");

    if (!success)
      return ToolResult::Error(-2, fmt::format("Failed to switch to {} disc", direction));

    JsonWriter w;
    w.StartObject();
    w.KeyString("status", "switched");
    w.KeyString("direction", direction);
    w.EndObject();
    return ToolResult{w.TakeOutput()};
  }

  return ToolResult::Error(-32602, "Must provide 'index' or 'direction' parameter");
}

static ToolResult HandleListDiscs([[maybe_unused]] const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  JsonWriter w;
  w.StartObject();
  w.KeyBool("has_playlist", System::HasMediaSubImages());

  if (System::HasMediaSubImages())
  {
    const u32 count = System::GetMediaSubImageCount();
    const u32 current = System::GetMediaSubImageIndex();

    w.KeyUint("count", count);
    w.KeyUint("current_index", current);

    w.Key("discs");
    w.StartArray();
    for (u32 i = 0; i < count; i++)
    {
      w.StartObject();
      w.KeyUint("index", i);
      w.KeyString("title", System::GetMediaSubImageTitle(i));
      w.KeyBool("current", i == current);
      w.EndObject();
    }
    w.EndArray();
  }

  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleListSaveStates([[maybe_unused]] const JsonValue& args)
{
  std::string serial;
  if (System::IsValid())
    serial = System::GetGameSerial();
  if (args.contains("serial") && args["serial"].is_string())
    serial = std::string(args["serial"].get_string());

  const std::vector<SaveStateInfo> states = System::GetAvailableSaveStates(serial);

  JsonWriter w;
  w.StartObject();
  w.KeyString("serial", serial);
  w.KeyUint("count", static_cast<u32>(states.size()));

  w.Key("states");
  w.StartArray();
  for (const auto& state : states)
  {
    w.StartObject();
    w.KeyString("path", state.path);
    w.KeyInt("slot", state.slot);
    w.KeyBool("global", state.global);
    w.KeyInt("timestamp", static_cast<s64>(state.timestamp));
    w.EndObject();
  }
  w.EndArray();

  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleSwapMemoryCards([[maybe_unused]] const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  System::SwapMemoryCards();

  JsonWriter w;
  w.StartObject();
  w.KeyString("status", "swapped");
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleBootGame(const JsonValue& args)
{
  if (System::IsValid())
    return ToolResult::Error(-1, "System is already running. Shut down first.");

  if (!args.contains("path") || !args["path"].is_string())
    return ToolResult::Error(-32602, "Missing 'path' parameter");

  const std::string path = std::string(args["path"].get_string());

  if (!FileSystem::FileExists(path.c_str()))
    return ToolResult::Error(-2, fmt::format("File not found: {}", path));

  SystemBootParameters params(path);

  if (args.contains("fast_boot") && args["fast_boot"].is_bool())
    params.override_fast_boot = args["fast_boot"].get_bool();
  if (args.contains("start_paused") && args["start_paused"].is_bool())
    params.override_start_paused = args["start_paused"].get_bool();
  if (args.contains("force_software") && args["force_software"].is_bool())
    params.force_software_renderer = args["force_software"].get_bool();

  Error error;
  if (!System::BootSystem(std::move(params), &error))
  {
    return ToolResult::Error(-2, fmt::format("Failed to boot game: {}", error.GetDescription()));
  }

  JsonWriter w;
  w.StartObject();
  w.KeyString("status", "booted");
  w.KeyString("path", path);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleShutdownSystem(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  bool save_resume_state = true;
  if (args.contains("save_resume_state") && args["save_resume_state"].is_bool())
    save_resume_state = args["save_resume_state"].get_bool();

  System::ShutdownSystem(save_resume_state);

  JsonWriter w;
  w.StartObject();
  w.KeyString("status", "shutdown");
  w.KeyBool("save_resume_state", save_resume_state);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}
// ---- CPU Debug Tool Handlers ----

// GPR: indices 0..34  (zero..ra, hi, lo, pc)
// COP0: indices 35..38
// GTE: indices 39..102

static ToolResult HandleReadRegisters(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  std::string group = "all";
  if (args.contains("group") && args["group"].is_string())
    group = std::string(args["group"].get_string());

  u32 start = 0, end = CPU::NUM_DEBUGGER_REGISTER_LIST_ENTRIES;
  if (group == "gpr")
  {
    start = 0;
    end = 35;
  }
  else if (group == "cop0")
  {
    start = 35;
    end = 39;
  }
  else if (group == "gte")
  {
    start = 39;
    end = 103;
  }

  JsonWriter w;
  w.StartObject();
  for (u32 i = start; i < end; i++)
  {
    const auto& entry = CPU::g_debugger_register_list[i];
    w.KeyString(entry.name, FormatHex32(*entry.value_ptr));
  }
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleWriteRegister(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (!args.contains("name") || !args["name"].is_string())
    return ToolResult::Error(-32602, "Missing 'name' parameter");
  if (!args.contains("value"))
    return ToolResult::Error(-32602, "Missing 'value' parameter");

  const std::string name = std::string(args["name"].get_string());
  const std::optional<u32> value = ParseAddress(args["value"]);
  if (!value.has_value())
    return ToolResult::Error(-32602, "Invalid value format");

  // Find the register by name.
  for (u32 i = 0; i < CPU::NUM_DEBUGGER_REGISTER_LIST_ENTRIES; i++)
  {
    if (StringUtil::EqualNoCase(CPU::g_debugger_register_list[i].name, name))
    {
      *CPU::g_debugger_register_list[i].value_ptr = value.value();

      // If writing pc, also update npc and clear icache.
      if (StringUtil::EqualNoCase(name, "pc"))
      {
        CPU::g_state.npc = value.value() + 4;
        CPU::ClearICache();
      }

      JsonWriter w;
      w.StartObject();
      w.KeyString("register", name);
      w.KeyString("value", FormatHex32(value.value()));
      w.EndObject();
      return ToolResult{w.TakeOutput()};
    }
  }

  return ToolResult::Error(-2, fmt::format("Unknown register: {}", name));
}

static ToolResult HandleDisassemble(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  std::optional<u32> address;
  if (args.contains("address"))
    address = ParseAddress(args["address"]);
  if (!address.has_value())
    return ToolResult::Error(-32602, "Missing or invalid 'address' parameter");

  u32 count = 20;
  if (args.contains("count") && args["count"].is_number())
    count = static_cast<u32>(args["count"].get_uint());

  JsonWriter w;
  w.StartArray();
  u32 addr = address.value();
  for (u32 i = 0; i < count; i++)
  {
    u32 instruction_bits = 0;
    if (!CPU::SafeReadMemoryWord(addr, &instruction_bits))
    {
      w.StartObject();
      w.KeyString("address", FormatHex32(addr));
      w.KeyString("bytes", "????????");
      w.KeyString("instruction", "<unreadable>");
      w.KeyString("comment", "");
      w.EndObject();
    }
    else
    {
      SmallString text;
      SmallString comment;
      CPU::DisassembleInstruction(&text, addr, instruction_bits);
      CPU::DisassembleInstructionComment(&comment, addr, instruction_bits);

      w.StartObject();
      w.KeyString("address", FormatHex32(addr));
      w.KeyString("bytes", FormatHex32(instruction_bits));
      w.KeyString("instruction", std::string(text.view()));
      w.KeyString("comment", std::string(comment.view()));
      w.EndObject();
    }
    addr += 4;
  }
  w.EndArray();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleStepInto([[maybe_unused]] const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  System::SingleStepCPU();

  // Return a GPR snapshot.
  return HandleReadRegisters(*JsonValue::Parse(R"({"group":"gpr"})"));
}

static ToolResult HandleStepOver([[maybe_unused]] const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  CPU::AddStepOverBreakpoint();
  System::PauseSystem(false);

  JsonWriter w;
  w.StartObject();
  w.KeyString("status", "step_over initiated, system resumed");
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleStepOut(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  u32 max_instructions = 10000;
  if (args.contains("max_instructions") && args["max_instructions"].is_number())
    max_instructions = static_cast<u32>(args["max_instructions"].get_uint());

  CPU::AddStepOutBreakpoint(max_instructions);
  System::PauseSystem(false);

  JsonWriter w;
  w.StartObject();
  w.KeyString("status", "step_out initiated, system resumed");
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandlePause([[maybe_unused]] const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  System::PauseSystem(true);

  JsonWriter w;
  w.StartObject();
  w.KeyString("status", "paused");
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleContinue([[maybe_unused]] const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  System::PauseSystem(false);

  JsonWriter w;
  w.StartObject();
  w.KeyString("status", "resumed");
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static void BroadcastSSEEvent(std::string_view event_name, std::string_view data_json)
{
  for (auto& client : s_sse_clients)
    client->SendSSEEvent(event_name, data_json);
}

static ToolResult HandleAddBreakpoint(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (!args.contains("type") || !args["type"].is_string())
    return ToolResult::Error(-32602, "Missing 'type' parameter");
  if (!args.contains("address"))
    return ToolResult::Error(-32602, "Missing 'address' parameter");

  const std::string type_str = std::string(args["type"].get_string());
  const std::optional<u32> address = ParseAddress(args["address"]);
  if (!address.has_value())
    return ToolResult::Error(-32602, "Invalid address format");

  CPU::BreakpointType bp_type;
  if (type_str == "execute")
    bp_type = CPU::BreakpointType::Execute;
  else if (type_str == "read")
    bp_type = CPU::BreakpointType::Read;
  else if (type_str == "write")
    bp_type = CPU::BreakpointType::Write;
  else
    return ToolResult::Error(-32602, fmt::format("Unknown breakpoint type: {}", type_str));

  if (!CPU::AddBreakpoint(bp_type, address.value()))
    return ToolResult::Error(-2, "Failed to add breakpoint (may already exist)");

  JsonWriter w;
  w.StartObject();
  w.KeyString("address", FormatHex32(address.value()));
  w.KeyString("type", type_str);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleRemoveBreakpoint(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (!args.contains("type") || !args["type"].is_string())
    return ToolResult::Error(-32602, "Missing 'type' parameter");
  if (!args.contains("address"))
    return ToolResult::Error(-32602, "Missing 'address' parameter");

  const std::string type_str = std::string(args["type"].get_string());
  const std::optional<u32> address = ParseAddress(args["address"]);
  if (!address.has_value())
    return ToolResult::Error(-32602, "Invalid address format");

  CPU::BreakpointType bp_type;
  if (type_str == "execute")
    bp_type = CPU::BreakpointType::Execute;
  else if (type_str == "read")
    bp_type = CPU::BreakpointType::Read;
  else if (type_str == "write")
    bp_type = CPU::BreakpointType::Write;
  else
    return ToolResult::Error(-32602, fmt::format("Unknown breakpoint type: {}", type_str));

  if (!CPU::RemoveBreakpoint(bp_type, address.value()))
    return ToolResult::Error(-2, "Breakpoint not found");

  JsonWriter w;
  w.StartObject();
  w.KeyString("status", "removed");
  w.KeyString("address", FormatHex32(address.value()));
  w.KeyString("type", type_str);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleListBreakpoints([[maybe_unused]] const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  const CPU::BreakpointList bps = CPU::CopyBreakpointList(false, true);
  JsonWriter w;
  w.StartArray();
  for (const auto& bp : bps)
  {
    const char* type_name = "unknown";
    switch (bp.type)
    {
      case CPU::BreakpointType::Execute:
        type_name = "execute";
        break;
      case CPU::BreakpointType::Read:
        type_name = "read";
        break;
      case CPU::BreakpointType::Write:
        type_name = "write";
        break;
      default:
        break;
    }

    w.StartObject();
    w.KeyString("address", FormatHex32(bp.address));
    w.KeyString("type", type_name);
    w.KeyBool("enabled", bp.enabled);
    w.KeyUint("hit_count", bp.hit_count);
    w.EndObject();
  }
  w.EndArray();
  return ToolResult{w.TakeOutput()};
}

// ---- Base64 Utilities ----

static constexpr const char BASE64_CHARS[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static std::string Base64Encode(const u8* data, size_t len)
{
  std::string result;
  result.reserve(((len + 2) / 3) * 4);

  for (size_t i = 0; i < len; i += 3)
  {
    const u32 b0 = data[i];
    const u32 b1 = (i + 1 < len) ? data[i + 1] : 0;
    const u32 b2 = (i + 2 < len) ? data[i + 2] : 0;
    const u32 triple = (b0 << 16) | (b1 << 8) | b2;

    result.push_back(BASE64_CHARS[(triple >> 18) & 0x3F]);
    result.push_back(BASE64_CHARS[(triple >> 12) & 0x3F]);
    result.push_back((i + 1 < len) ? BASE64_CHARS[(triple >> 6) & 0x3F] : '=');
    result.push_back((i + 2 < len) ? BASE64_CHARS[triple & 0x3F] : '=');
  }

  return result;
}

static std::vector<u8> Base64Decode(std::string_view encoded)
{
  // Build reverse lookup table.
  static constexpr auto MakeDecodingTable = []() {
    std::array<u8, 256> table{};
    table.fill(0xFF);
    for (u8 i = 0; i < 64; i++)
      table[static_cast<u8>(BASE64_CHARS[i])] = i;
    return table;
  };
  static constexpr auto DECODE_TABLE = MakeDecodingTable();

  std::vector<u8> result;
  result.reserve((encoded.size() / 4) * 3);

  u32 accum = 0;
  int bits = 0;
  for (const char ch : encoded)
  {
    if (ch == '=' || ch == '\n' || ch == '\r' || ch == ' ')
      continue;

    const u8 val = DECODE_TABLE[static_cast<u8>(ch)];
    if (val == 0xFF)
      continue; // Skip invalid characters.

    accum = (accum << 6) | val;
    bits += 6;
    if (bits >= 8)
    {
      bits -= 8;
      result.push_back(static_cast<u8>((accum >> bits) & 0xFF));
    }
  }

  return result;
}

// ---- Memory Tool Handlers ----

static ToolResult HandleReadMemory(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (!args.contains("address"))
    return ToolResult::Error(-32602, "Missing 'address' parameter");
  if (!args.contains("size") || !args["size"].is_number())
    return ToolResult::Error(-32602, "Missing or invalid 'size' parameter");

  const std::optional<u32> address = ParseAddress(args["address"]);
  if (!address.has_value())
    return ToolResult::Error(-32602, "Invalid address format");

  const u32 size = static_cast<u32>(args["size"].get_uint());
  static constexpr u32 MAX_READ_SIZE = 1048576; // 1MB
  if (size == 0 || size > MAX_READ_SIZE)
    return ToolResult::Error(-32602, "Size must be between 1 and 1048576 (1MB)");

  std::string format = "hex";
  if (args.contains("format") && args["format"].is_string())
    format = std::string(args["format"].get_string());

  std::vector<u8> buffer(size);
  if (!CPU::SafeReadMemoryBytes(address.value(), buffer.data(), size))
    return ToolResult::Error(-2, "Failed to read memory at specified address");

  std::string data_str;
  if (format == "base64")
    data_str = Base64Encode(buffer.data(), buffer.size());
  else
    data_str = StringUtil::EncodeHex(buffer.data(), buffer.size());

  JsonWriter w;
  w.StartObject();
  w.KeyString("address", FormatHex32(address.value()));
  w.KeyUint("size", size);
  w.KeyString("data", data_str);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleWriteMemory(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (!args.contains("address"))
    return ToolResult::Error(-32602, "Missing 'address' parameter");
  if (!args.contains("data") || !args["data"].is_string())
    return ToolResult::Error(-32602, "Missing or invalid 'data' parameter");

  const std::optional<u32> address = ParseAddress(args["address"]);
  if (!address.has_value())
    return ToolResult::Error(-32602, "Invalid address format");

  const std::string data_str = std::string(args["data"].get_string());

  std::string format = "hex";
  if (args.contains("format") && args["format"].is_string())
    format = std::string(args["format"].get_string());

  std::vector<u8> buffer;
  if (format == "base64")
  {
    buffer = Base64Decode(data_str);
  }
  else
  {
    const std::optional<std::vector<u8>> decoded = StringUtil::DecodeHex(data_str);
    if (!decoded.has_value() || decoded->empty())
      return ToolResult::Error(-32602, "Invalid hex data string");
    buffer = decoded.value();
  }

  if (buffer.empty())
    return ToolResult::Error(-32602, "Decoded data is empty");

  if (!CPU::SafeWriteMemoryBytes(address.value(), std::span<const u8>(buffer)))
    return ToolResult::Error(-2, "Failed to write memory at specified address");

  JsonWriter w;
  w.StartObject();
  w.KeyString("address", FormatHex32(address.value()));
  w.KeyUint("bytes_written", static_cast<u32>(buffer.size()));
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleSearchMemory(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (!args.contains("pattern") || !args["pattern"].is_string())
    return ToolResult::Error(-32602, "Missing or invalid 'pattern' parameter");

  u32 start_address = 0;
  if (args.contains("start"))
  {
    const std::optional<u32> parsed_start = ParseAddress(args["start"]);
    if (!parsed_start.has_value())
      return ToolResult::Error(-32602, "Invalid start address format");
    start_address = parsed_start.value();
  }

  const std::string pattern_hex = std::string(args["pattern"].get_string());
  const std::optional<std::vector<u8>> pattern_opt = StringUtil::DecodeHex(pattern_hex);
  if (!pattern_opt.has_value() || pattern_opt->empty())
    return ToolResult::Error(-32602, "Invalid pattern hex string");

  const std::vector<u8>& pattern = pattern_opt.value();
  std::vector<u8> mask;

  if (args.contains("mask") && args["mask"].is_string())
  {
    const std::string mask_hex = std::string(args["mask"].get_string());
    const std::optional<std::vector<u8>> mask_opt = StringUtil::DecodeHex(mask_hex);
    if (!mask_opt.has_value() || mask_opt->size() != pattern.size())
      return ToolResult::Error(-32602, "Invalid mask hex string or size mismatch with pattern");
    mask = mask_opt.value();
  }
  else
  {
    // Default mask: all 0xFF (exact match).
    mask.resize(pattern.size(), 0xFF);
  }

  static constexpr size_t MAX_MATCHES = 100;
  std::vector<std::string> matches;
  PhysicalMemoryAddress search_addr = start_address;

  while (matches.size() < MAX_MATCHES)
  {
    const std::optional<PhysicalMemoryAddress> result =
      Bus::SearchMemory(search_addr, pattern.data(), mask.data(), static_cast<u32>(pattern.size()));
    if (!result.has_value())
      break;

    matches.push_back(FormatHex32(result.value()));
    search_addr = result.value() + 1;
  }

  JsonWriter w;
  w.StartObject();
  w.Key("matches");
  w.StartArray();
  for (const auto& m : matches)
    w.String(m);
  w.EndArray();
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleDumpRam(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (!args.contains("path") || !args["path"].is_string())
    return ToolResult::Error(-32602, "Missing required 'path' parameter");

  const std::string path = std::string(args["path"].get_string());
  const u32 ram_size = Bus::g_ram_size;

  Error error;
  if (!FileSystem::WriteBinaryFile(path.c_str(), Bus::g_ram, ram_size, &error))
  {
    return ToolResult::Error(-2, fmt::format("Failed to write file: {}", error.GetDescription()));
  }

  JsonWriter w;
  w.StartObject();
  w.KeyString("path", path);
  w.KeyUint("size", ram_size);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

// ---- Hardware State Tool Handlers ----

static ToolResult HandleGetGpuState([[maybe_unused]] const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  // Read GPUSTAT via the public ReadRegister interface (offset 0x04).
  const u32 gpustat = g_gpu.ReadRegister(0x04);

  // Get display resolution.
  const auto [display_width, display_height] = g_gpu.GetFullDisplayResolution();

  // Read public CRTC/display state via inline accessors.
  const bool interlaced = g_gpu.IsInterlacedDisplayEnabled();
  const bool pal_mode = g_gpu.IsInPALMode();
  const bool display_disabled = g_gpu.IsDisplayDisabled();
  const u8 resolution_scale = g_gpu_settings.gpu_resolution_scale;

  JsonWriter w;
  w.StartObject();
  w.KeyString("gpustat", FormatHex32(gpustat));
  w.KeyUint("display_width", display_width);
  w.KeyUint("display_height", display_height);
  w.KeyBool("interlaced", interlaced);
  w.KeyBool("pal_mode", pal_mode);
  w.KeyBool("display_disabled", display_disabled);
  w.KeyUint("resolution_scale", resolution_scale);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleGetSpuState([[maybe_unused]] const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  // Read SPU registers via the public ReadRegister interface.
  // SPU_BASE = 0x1F801C00, register offsets are relative to SPU_BASE.
  const u16 spucnt = SPU::ReadRegister(0x1F801DAA - 0x1F801C00);        // Control register
  const u16 spustat = SPU::ReadRegister(0x1F801DAE - 0x1F801C00);       // Status register
  const u16 main_vol_left = SPU::ReadRegister(0x1F801D80 - 0x1F801C00); // Main volume left
  const u16 main_vol_right = SPU::ReadRegister(0x1F801D82 - 0x1F801C00); // Main volume right
  const u16 cd_vol_left = SPU::ReadRegister(0x1F801DB0 - 0x1F801C00);   // CD audio volume left
  const u16 cd_vol_right = SPU::ReadRegister(0x1F801DB2 - 0x1F801C00);  // CD audio volume right
  const u16 reverb_vol_left = SPU::ReadRegister(0x1F801D84 - 0x1F801C00);  // Reverb output left
  const u16 reverb_vol_right = SPU::ReadRegister(0x1F801D86 - 0x1F801C00); // Reverb output right
  const u16 transfer_ctrl = SPU::ReadRegister(0x1F801DAC - 0x1F801C00);    // Transfer control

  JsonWriter w;
  w.StartObject();
  w.KeyString("spucnt", FormatHex32(spucnt));
  w.KeyString("spustat", FormatHex32(spustat));
  w.KeyInt("main_volume_left", static_cast<s16>(main_vol_left));
  w.KeyInt("main_volume_right", static_cast<s16>(main_vol_right));
  w.KeyInt("cd_audio_volume_left", static_cast<s16>(cd_vol_left));
  w.KeyInt("cd_audio_volume_right", static_cast<s16>(cd_vol_right));
  w.KeyInt("reverb_output_volume_left", static_cast<s16>(reverb_vol_left));
  w.KeyInt("reverb_output_volume_right", static_cast<s16>(reverb_vol_right));
  w.KeyString("transfer_control", FormatHex32(transfer_ctrl));
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleGetCdromState([[maybe_unused]] const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  const bool has_media = CDROM::HasMedia();
  const DiscRegion disc_region = CDROM::GetDiscRegion();
  const bool is_ps1_disc = has_media ? CDROM::IsMediaPS1Disc() : false;
  const bool is_audio_cd = has_media ? CDROM::IsMediaAudioCD() : false;
  const u32 current_sub_image = CDROM::GetCurrentSubImage();

  // Read CDROM status register (index port at offset 0).
  const u8 status_reg = CDROM::ReadRegister(0);

  JsonWriter w;
  w.StartObject();
  w.KeyBool("has_media", has_media);
  w.KeyString("disc_region", Settings::GetDiscRegionName(disc_region));
  w.KeyBool("is_ps1_disc", is_ps1_disc);
  w.KeyBool("is_audio_cd", is_audio_cd);
  w.KeyUint("current_sub_image", current_sub_image);
  w.KeyString("status_register", FormatHex32(status_reg));

  // If we have media, try to get track/position info from the CDImage.
  if (has_media)
  {
    const CDImage* media = CDROM::GetMedia();
    if (media)
    {
      w.KeyUint("current_lba", media->GetPositionOnDisc());
      w.KeyUint("current_track", media->GetTrackNumber());
      w.KeyUint("track_count", media->GetTrackCount());
      w.KeyString("media_path", std::string(CDROM::GetMediaPath()));
    }
  }

  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleGetDmaState([[maybe_unused]] const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  static constexpr const char* CHANNEL_NAMES[DMA::NUM_CHANNELS] = {"MDECin", "MDECout", "GPU",  "CDROM",
                                                                    "SPU",    "PIO",     "OTC"};

  JsonWriter w;
  w.StartObject();
  w.Key("channels");
  w.StartArray();
  for (u32 i = 0; i < DMA::NUM_CHANNELS; i++)
  {
    // Each channel has registers at offset (i * 0x10) + {0x00, 0x04, 0x08}.
    const u32 base_addr = DMA::ReadRegister(i * 0x10 + 0x00);     // Base address
    const u32 block_ctrl = DMA::ReadRegister(i * 0x10 + 0x04);    // Block control
    const u32 channel_ctrl = DMA::ReadRegister(i * 0x10 + 0x08);  // Channel control

    const bool enable_busy = (channel_ctrl & (1u << 24)) != 0;

    w.StartObject();
    w.KeyUint("channel", i);
    w.KeyString("name", CHANNEL_NAMES[i]);
    w.KeyString("base_address", FormatHex32(base_addr));
    w.KeyString("block_control", FormatHex32(block_ctrl));
    w.KeyString("channel_control", FormatHex32(channel_ctrl));
    w.KeyBool("active", enable_busy);
    w.EndObject();
  }
  w.EndArray();

  // Also read DPCR (DMA control register) and DICR (DMA interrupt register).
  const u32 dpcr = DMA::ReadRegister(0x70);
  const u32 dicr = DMA::ReadRegister(0x74);

  w.KeyString("dpcr", FormatHex32(dpcr));
  w.KeyString("dicr", FormatHex32(dicr));
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleGetTimersState([[maybe_unused]] const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  static constexpr const char* TIMER_NAMES[3] = {"Dotclock", "HBlank", "Sysclk/8"};

  JsonWriter w;
  w.StartObject();
  w.Key("timers");
  w.StartArray();
  for (u32 i = 0; i < 3; i++)
  {
    // Each timer has registers at offset (i * 0x10) + {0x00, 0x04, 0x08}.
    // Reading mode (offset 0x04) has a side effect of clearing reached flags,
    // so we read counter and target only, and use the public accessors for mode info.
    const u32 counter = Timers::ReadRegister(i * 0x10 + 0x00);
    const u32 target = Timers::ReadRegister(i * 0x10 + 0x08);
    const bool external_clock = Timers::IsUsingExternalClock(i);
    const bool sync_enabled = Timers::IsSyncEnabled(i);
    const bool external_irq_enabled = Timers::IsExternalIRQEnabled(i);

    w.StartObject();
    w.KeyUint("timer", i);
    w.KeyString("name", TIMER_NAMES[i]);
    w.KeyUint("counter", counter);
    w.KeyUint("target", target);
    w.KeyBool("using_external_clock", external_clock);
    w.KeyBool("sync_enabled", sync_enabled);
    w.KeyBool("external_irq_enabled", external_irq_enabled);
    w.EndObject();
  }
  w.EndArray();
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleDumpVram(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (!args.contains("path") || !args["path"].is_string())
    return ToolResult::Error(-32602, "Missing required 'path' parameter");

  const std::string path = std::string(args["path"].get_string());

  std::string format = "png";
  if (args.contains("format") && args["format"].is_string())
    format = std::string(args["format"].get_string());

  static constexpr u32 VRAM_W = VRAM_WIDTH;
  static constexpr u32 VRAM_H = VRAM_HEIGHT;

  Error error;

  if (format == "bin")
  {
    // Save raw 16-bit VRAM data.
    if (!FileSystem::WriteBinaryFile(path.c_str(), reinterpret_cast<const u8*>(g_vram),
                                     VRAM_W * VRAM_H * sizeof(u16), &error))
    {
      return ToolResult::Error(-2, fmt::format("Failed to write file: {}", error.GetDescription()));
    }

    JsonWriter w;
    w.StartObject();
    w.KeyString("path", path);
    w.KeyString("format", "bin");
    w.KeyUint("width", VRAM_W);
    w.KeyUint("height", VRAM_H);
    w.KeyUint("size", VRAM_W * VRAM_H * sizeof(u16));
    w.EndObject();
    return ToolResult{w.TakeOutput()};
  }
  else if (format == "png")
  {
    // Convert 16-bit VRAM (1555 ABGR) to RGBA8888, then save as PNG.
    Image img(VRAM_W, VRAM_H, ImageFormat::RGBA8);
    u8* dst_pixels = img.GetPixels();
    const u32 dst_pitch = img.GetPitch();

    for (u32 y = 0; y < VRAM_H; y++)
    {
      u8* dst_row = dst_pixels + y * dst_pitch;
      for (u32 x = 0; x < VRAM_W; x++)
      {
        const u16 pixel = g_vram[y * VRAM_W + x];
        const u8 r5 = static_cast<u8>(pixel & 0x1F);
        const u8 g5 = static_cast<u8>((pixel >> 5) & 0x1F);
        const u8 b5 = static_cast<u8>((pixel >> 10) & 0x1F);
        dst_row[x * 4 + 0] = (r5 << 3) | (r5 >> 2);
        dst_row[x * 4 + 1] = (g5 << 3) | (g5 >> 2);
        dst_row[x * 4 + 2] = (b5 << 3) | (b5 >> 2);
        dst_row[x * 4 + 3] = 0xFF;
      }
    }

    if (!img.SaveToFile(path.c_str(), Image::DEFAULT_SAVE_QUALITY, &error))
    {
      return ToolResult::Error(-2, fmt::format("Failed to save PNG: {}", error.GetDescription()));
    }

    JsonWriter w;
    w.StartObject();
    w.KeyString("path", path);
    w.KeyString("format", "png");
    w.KeyUint("width", VRAM_W);
    w.KeyUint("height", VRAM_H);
    w.EndObject();
    return ToolResult{w.TakeOutput()};
  }
  else
  {
    return ToolResult::Error(-32602, "Invalid format. Use 'png' or 'bin'.");
  }
}

static ToolResult HandleDumpSpuRam(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (!args.contains("path") || !args["path"].is_string())
    return ToolResult::Error(-32602, "Missing required 'path' parameter");

  const std::string path = std::string(args["path"].get_string());
  const auto& spu_ram = SPU::GetRAM();

  Error error;
  if (!FileSystem::WriteBinaryFile(path.c_str(), spu_ram.data(), spu_ram.size(), &error))
  {
    return ToolResult::Error(-2, fmt::format("Failed to write file: {}", error.GetDescription()));
  }

  JsonWriter w;
  w.StartObject();
  w.KeyString("path", path);
  w.KeyUint("size", SPU::RAM_SIZE);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

// ---- System Control Tool Handlers ----

static const char* MapSystemStateToString(System::State state)
{
  switch (state)
  {
    case System::State::Shutdown:
      return "shutdown";
    case System::State::Starting:
      return "starting";
    case System::State::Running:
      return "running";
    case System::State::Paused:
      return "paused";
    case System::State::Stopping:
      return "stopping";
    default:
      return "unknown";
  }
}

static ToolResult HandleGetStatus([[maybe_unused]] const JsonValue& args)
{
  const System::State state = System::GetState();

  JsonWriter w;
  w.StartObject();
  w.KeyString("state", MapSystemStateToString(state));

  if (System::IsValid())
  {
    w.KeyString("game_serial", System::GetGameSerial());
    w.KeyString("game_title", System::GetGameTitle());
    w.KeyUint("frame_number", System::GetFrameNumber());
    w.KeyUint("internal_frame_number", System::GetInternalFrameNumber());
    w.KeyUint("global_tick_counter", System::GetGlobalTickCounter());
  }
  else
  {
    w.KeyString("game_serial", "");
    w.KeyString("game_title", "");
    w.KeyUint("frame_number", 0);
    w.KeyUint("internal_frame_number", 0);
    w.KeyUint("global_tick_counter", 0);
  }

  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleReset([[maybe_unused]] const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  System::ResetSystem();

  JsonWriter w;
  w.StartObject();
  w.KeyString("status", "reset");
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleSaveState(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (!args.contains("slot") || !args["slot"].is_number_integer())
    return ToolResult::Error(-32602, "Missing or invalid 'slot' parameter (integer 1-10)");

  const s32 slot = static_cast<s32>(args["slot"].get_int());
  if (slot < 1 || slot > 10)
    return ToolResult::Error(-32602, "Slot must be between 1 and 10");

  System::SaveStateToSlot(false, slot);

  JsonWriter w;
  w.StartObject();
  w.KeyString("status", "saved");
  w.KeyInt("slot", slot);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleLoadState(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (!args.contains("slot") || !args["slot"].is_number_integer())
    return ToolResult::Error(-32602, "Missing or invalid 'slot' parameter (integer 1-10)");

  const s32 slot = static_cast<s32>(args["slot"].get_int());
  if (slot < 1 || slot > 10)
    return ToolResult::Error(-32602, "Slot must be between 1 and 10");

  System::LoadStateFromSlot(false, slot);

  JsonWriter w;
  w.StartObject();
  w.KeyString("status", "loaded");
  w.KeyInt("slot", slot);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleFrameStep(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  // count parameter is accepted but we can only step one frame at a time.
  // The system will pause after 1 frame; the client can call frame_step again for more.
  System::DoFrameStep();

  JsonWriter w;
  w.StartObject();
  w.KeyString("status", "stepped");
  w.KeyUint("frame_number", System::GetFrameNumber());
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

// ---- VRAM Watch Tool Handlers ----

static ToolResult HandleWatchVramWrite(const JsonValue& args)
{
  if (!args.contains("x") || !args.contains("y") || !args.contains("width") || !args.contains("height"))
    return ToolResult::Error(-32602, "Missing required parameters: x, y, width, height");

  auto x_opt = ParseAddress(args["x"]);
  auto y_opt = ParseAddress(args["y"]);
  auto w_opt = ParseAddress(args["width"]);
  auto h_opt = ParseAddress(args["height"]);

  if (!x_opt || !y_opt || !w_opt || !h_opt)
    return ToolResult::Error(-32602, "Invalid parameter values");

  if (*x_opt + *w_opt > 1024 || *y_opt + *h_opt > 512)
    return ToolResult::Error(-32602,
      fmt::format("VRAM watch region ({},{})+{}x{} exceeds VRAM bounds (1024x512)",
                  *x_opt, *y_opt, *w_opt, *h_opt));

  VRAMWatch watch;
  watch.id = s_next_vram_watch_id++;
  watch.x = static_cast<u16>(*x_opt);
  watch.y = static_cast<u16>(*y_opt);
  watch.width = static_cast<u16>(*w_opt);
  watch.height = static_cast<u16>(*h_opt);
  s_vram_watches.push_back(watch);

  DEV_LOG("VRAM watch #{} added: ({},{}) {}x{}", watch.id, watch.x, watch.y, watch.width, watch.height);

  JsonWriter w;
  w.StartObject();
  w.KeyUint("id", watch.id);
  w.KeyUint("x", watch.x);
  w.KeyUint("y", watch.y);
  w.KeyUint("width", watch.width);
  w.KeyUint("height", watch.height);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleRemoveVramWatch(const JsonValue& args)
{
  if (!args.contains("id") || !args["id"].is_number())
    return ToolResult::Error(-32602, "Missing required parameter: id");

  const u32 id = static_cast<u32>(args["id"].get_uint());
  auto it = std::find_if(s_vram_watches.begin(), s_vram_watches.end(),
                          [id](const VRAMWatch& w) { return w.id == id; });

  if (it == s_vram_watches.end())
    return ToolResult::Error(-2, fmt::format("VRAM watch #{} not found", id));

  s_vram_watches.erase(it);
  JsonWriter w;
  w.StartObject();
  w.KeyString("status", "removed");
  w.KeyUint("id", id);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleListVramWatches([[maybe_unused]] const JsonValue& args)
{
  JsonWriter w;
  w.StartObject();
  w.Key("watches");
  w.StartArray();
  for (const auto& vw : s_vram_watches)
  {
    w.StartObject();
    w.KeyUint("id", vw.id);
    w.KeyUint("x", vw.x);
    w.KeyUint("y", vw.y);
    w.KeyUint("width", vw.width);
    w.KeyUint("height", vw.height);
    w.EndObject();
  }
  w.EndArray();
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleGetVramWatchLastHit([[maybe_unused]] const JsonValue& args)
{
  if (!s_vram_watch_hit_pending)
    return ToolResult::Error(-2, "No VRAM watch hit recorded");

  static constexpr const char* REG_NAMES[32] = {
    "zero","at","v0","v1","a0","a1","a2","a3",
    "t0","t1","t2","t3","t4","t5","t6","t7",
    "s0","s1","s2","s3","s4","s5","s6","s7",
    "t8","t9","k0","k1","gp","sp","fp","ra"
  };

  JsonWriter w;
  w.StartObject();
  w.KeyString("pc", fmt::format("0x{:08X}", s_vram_watch_hit_pc));
  w.KeyUint("x", s_vram_watch_hit_x);
  w.KeyUint("y", s_vram_watch_hit_y);
  w.KeyUint("width", s_vram_watch_hit_w);
  w.KeyUint("height", s_vram_watch_hit_h);

  w.Key("regs");
  w.StartObject();
  for (int i = 0; i < 32; i++)
    w.KeyString(REG_NAMES[i], fmt::format("0x{:08X}", s_vram_watch_hit_regs[i]));
  w.EndObject();

  w.Key("stack");
  w.StartArray();
  const u32 base_sp = s_vram_watch_hit_sp;
  for (u32 i = 0; i < 64; i++)
  {
    w.StartObject();
    w.KeyString("addr", fmt::format("0x{:08X}", base_sp + i * 4));
    w.KeyString("val", fmt::format("0x{:08X}", s_vram_watch_hit_stack[i]));
    w.EndObject();
  }
  w.EndArray();

  w.EndObject();
  s_vram_watch_hit_pending = false;
  return ToolResult{w.TakeOutput()};
}

// ---- Batch 1: CPU Debug Enhancement Tool Handlers ----

static ToolResult HandleGetCop0State([[maybe_unused]] const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  const auto& cop0 = CPU::g_state.cop0_regs;

  // Decode exception code name
  static constexpr const char* exception_names[] = {
    "INT", "MOD", "TLBL", "TLBS", "AdEL", "AdES", "IBE", "DBE",
    "Syscall", "BP", "RI", "CpU", "Ov"
  };
  const u8 excode = static_cast<u8>(cop0.cause.Excode.GetValue());
  const char* excode_name = (excode < std::size(exception_names)) ? exception_names[excode] : "Unknown";

  JsonWriter w;
  w.StartObject();

  // sr decoded
  w.Key("sr");
  w.StartObject();
  w.KeyString("bits", FormatHex32(cop0.sr.bits));
  w.KeyBool("IEc", (bool)cop0.sr.IEc);
  w.KeyBool("KUc", (bool)cop0.sr.KUc);
  w.KeyBool("IEp", (bool)cop0.sr.IEp);
  w.KeyBool("KUp", (bool)cop0.sr.KUp);
  w.KeyBool("IEo", (bool)cop0.sr.IEo);
  w.KeyBool("KUo", (bool)cop0.sr.KUo);
  w.KeyUint("Im", cop0.sr.Im.GetValue());
  w.KeyBool("Isc", (bool)cop0.sr.Isc);
  w.KeyBool("Swc", (bool)cop0.sr.Swc);
  w.KeyBool("BEV", (bool)cop0.sr.BEV);
  w.KeyBool("CU0", (bool)cop0.sr.CU0);
  w.KeyBool("CE1", (bool)cop0.sr.CE1);
  w.KeyBool("CE2", (bool)cop0.sr.CE2);
  w.KeyBool("CE3", (bool)cop0.sr.CE3);
  w.EndObject();

  // cause decoded
  w.Key("cause");
  w.StartObject();
  w.KeyString("bits", FormatHex32(cop0.cause.bits));
  w.KeyUint("Excode", excode);
  w.KeyString("Excode_name", excode_name);
  w.KeyUint("Ip", cop0.cause.Ip.GetValue());
  w.KeyUint("CE", cop0.cause.CE.GetValue());
  w.KeyBool("BD", (bool)cop0.cause.BD);
  w.KeyBool("BT", (bool)cop0.cause.BT);
  w.EndObject();

  // dcic decoded
  w.Key("dcic");
  w.StartObject();
  w.KeyString("bits", FormatHex32(cop0.dcic.bits));
  w.KeyBool("status_any_break", (bool)cop0.dcic.status_any_break);
  w.KeyBool("status_bpc_code_break", (bool)cop0.dcic.status_bpc_code_break);
  w.KeyBool("status_bda_data_break", (bool)cop0.dcic.status_bda_data_break);
  w.KeyBool("execution_breakpoint_enable", (bool)cop0.dcic.execution_breakpoint_enable);
  w.KeyBool("data_access_breakpoint", (bool)cop0.dcic.data_access_breakpoint);
  w.KeyBool("break_on_data_read", (bool)cop0.dcic.break_on_data_read);
  w.KeyBool("break_on_data_write", (bool)cop0.dcic.break_on_data_write);
  w.KeyBool("master_enable_break", (bool)cop0.dcic.master_enable_break);
  w.KeyBool("super_master_enable_1", (bool)cop0.dcic.super_master_enable_1);
  w.EndObject();

  // Top-level register values
  w.KeyString("EPC", FormatHex32(cop0.EPC));
  w.KeyString("BadVaddr", FormatHex32(cop0.BadVaddr));
  w.KeyString("BPC", FormatHex32(cop0.BPC));
  w.KeyString("BDA", FormatHex32(cop0.BDA));
  w.KeyString("BPCM", FormatHex32(cop0.BPCM));
  w.KeyString("BDAM", FormatHex32(cop0.BDAM));
  w.KeyString("TAR", FormatHex32(cop0.TAR));
  w.KeyString("PRID", FormatHex32(cop0.PRID));

  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleGetCpuExecutionState([[maybe_unused]] const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  const auto& state = CPU::g_state;

  JsonWriter w;
  w.StartObject();
  w.KeyString("pc", FormatHex32(state.pc));
  w.KeyString("npc", FormatHex32(state.npc));
  w.KeyString("current_instruction_pc", FormatHex32(state.current_instruction_pc));
  w.KeyString("current_instruction", FormatHex32(state.current_instruction.bits));
  w.KeyBool("in_branch_delay_slot", state.current_instruction_in_branch_delay_slot);
  w.KeyBool("branch_was_taken", state.current_instruction_was_branch_taken);
  w.KeyBool("next_is_branch_delay_slot", state.next_instruction_is_branch_delay_slot);
  w.KeyBool("exception_raised", state.exception_raised);
  w.KeyBool("bus_error", state.bus_error);
  w.KeyInt("pending_ticks", state.pending_ticks);
  w.KeyInt("downcount", state.downcount);
  w.KeyUint("load_delay_reg", static_cast<u32>(state.load_delay_reg));
  w.KeyString("load_delay_value", FormatHex32(state.load_delay_value));
  w.KeyUint("next_load_delay_reg", static_cast<u32>(state.next_load_delay_reg));
  w.KeyString("next_load_delay_value", FormatHex32(state.next_load_delay_value));
  w.KeyUint("gte_completion_tick", state.gte_completion_tick);
  w.KeyUint("muldiv_completion_tick", state.muldiv_completion_tick);
  w.KeyBool("using_interpreter", state.using_interpreter);
  w.KeyBool("using_debug_dispatcher", state.using_debug_dispatcher);
  w.KeyString("cache_control", FormatHex32(state.cache_control.bits));
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleStartTrace([[maybe_unused]] const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (CPU::IsTraceEnabled())
    return ToolResult::Error(-2, "Trace already active");

  CPU::StartTrace();

  JsonWriter w;
  w.StartObject();
  w.KeyBool("success", true);
  w.KeyBool("tracing", true);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleStopTrace([[maybe_unused]] const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (!CPU::IsTraceEnabled())
    return ToolResult::Error(-2, "No trace active");

  CPU::StopTrace();

  JsonWriter w;
  w.StartObject();
  w.KeyBool("success", true);
  w.KeyBool("tracing", false);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleGetTraceStatus([[maybe_unused]] const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  JsonWriter w;
  w.StartObject();
  w.KeyBool("tracing", CPU::IsTraceEnabled());
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleEnableBreakpoint(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (!args.contains("address"))
    return ToolResult::Error(-32602, "Missing 'address' parameter");

  const auto addr = ParseAddress(args["address"]);
  if (!addr.has_value())
    return ToolResult::Error(-32602, "Invalid address");

  const auto bp_type = ParseBreakpointType(args);
  if (!bp_type.has_value())
    return ToolResult::Error(-32602, "Invalid breakpoint type (must be 'execute', 'read', or 'write')");

  if (!CPU::SetBreakpointEnabled(bp_type.value(), addr.value(), true))
    return ToolResult::Error(-2, "Breakpoint not found at specified address");

  const char* type_name = CPU::GetBreakpointTypeName(bp_type.value());
  JsonWriter w;
  w.StartObject();
  w.KeyBool("success", true);
  w.KeyString("address", FormatHex32(addr.value()));
  w.KeyString("type", type_name);
  w.KeyBool("enabled", true);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleDisableBreakpoint(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (!args.contains("address"))
    return ToolResult::Error(-32602, "Missing 'address' parameter");

  const auto addr = ParseAddress(args["address"]);
  if (!addr.has_value())
    return ToolResult::Error(-32602, "Invalid address");

  const auto bp_type = ParseBreakpointType(args);
  if (!bp_type.has_value())
    return ToolResult::Error(-32602, "Invalid breakpoint type (must be 'execute', 'read', or 'write')");

  if (!CPU::SetBreakpointEnabled(bp_type.value(), addr.value(), false))
    return ToolResult::Error(-2, "Breakpoint not found at specified address");

  const char* type_name = CPU::GetBreakpointTypeName(bp_type.value());
  JsonWriter w;
  w.StartObject();
  w.KeyBool("success", true);
  w.KeyString("address", FormatHex32(addr.value()));
  w.KeyString("type", type_name);
  w.KeyBool("enabled", false);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleClearBreakpoints([[maybe_unused]] const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  CPU::ClearBreakpoints();

  JsonWriter w;
  w.StartObject();
  w.KeyBool("success", true);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

// ---- Batch 2: GTE Tool Handlers ----

static ToolResult HandleGetGteRegisters([[maybe_unused]] const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  // Data registers 0-31
  static constexpr const char* data_reg_names[] = {
    "VXY0", "VZ0", "VXY1", "VZ1", "VXY2", "VZ2", "RGBC", "OTZ",
    "IR0", "IR1", "IR2", "IR3", "SXY0", "SXY1", "SXY2", "SXYP",
    "SZ0", "SZ1", "SZ2", "SZ3", "RGB0", "RGB1", "RGB2", "RES1",
    "MAC0", "MAC1", "MAC2", "MAC3", "IRGB", "ORGB", "LZCS", "LZCR"
  };

  // Control registers 32-63
  static constexpr const char* ctrl_reg_names[] = {
    "RT11RT12", "RT13RT21", "RT22RT23", "RT31RT32", "RT33", "TRX", "TRY", "TRZ",
    "L11L12", "L13L21", "L22L23", "L31L32", "L33", "RBK", "GBK", "BBK",
    "LR1LR2", "LR3LG1", "LG2LG3", "LB1LB2", "LB3", "RFC", "GFC", "BFC",
    "OFX", "OFY", "H", "DQA", "DQB", "ZSF3", "ZSF4", "FLAG"
  };

  JsonWriter w;
  w.StartObject();

  w.Key("data_registers");
  w.StartObject();
  for (u32 i = 0; i < 32; i++)
    w.KeyString(data_reg_names[i], FormatHex32(GTE::ReadRegister(i)));
  w.EndObject();

  w.Key("control_registers");
  w.StartObject();
  for (u32 i = 0; i < 32; i++)
    w.KeyString(ctrl_reg_names[i], FormatHex32(GTE::ReadRegister(i + 32)));
  w.EndObject();

  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleSetGteRegister(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (!args.contains("index") || !args["index"].is_number_unsigned())
    return ToolResult::Error(-32602, "Missing 'index' parameter (0-63)");
  if (!args.contains("value"))
    return ToolResult::Error(-32602, "Missing 'value' parameter");

  const u32 index = static_cast<u32>(args["index"].get_uint());
  if (index >= 64)
    return ToolResult::Error(-32602, "Register index must be 0-63");

  const auto value = ParseAddress(args["value"]);
  if (!value.has_value())
    return ToolResult::Error(-32602, "Invalid value");

  GTE::WriteRegister(index, value.value());

  JsonWriter w;
  w.StartObject();
  w.KeyBool("success", true);
  w.KeyUint("index", index);
  w.KeyString("value", FormatHex32(value.value()));
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

// ---- Batch 3: Hardware Debug Tool Handlers ----

static ToolResult HandleGetInterruptState([[maybe_unused]] const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  const u32 status = InterruptController::ReadRegister(0x00);
  const u32 mask = InterruptController::ReadRegister(0x04);

  static constexpr const char* irq_names[] = {
    "VBLANK", "GPU", "CDROM", "DMA", "TMR0", "TMR1", "TMR2", "PAD", "SIO", "SPU", "IRQ10"
  };
  static_assert(countof(irq_names) == InterruptController::NUM_IRQS);

  JsonWriter w;
  w.StartObject();
  w.KeyString("status_register", FormatHex32(status));
  w.KeyString("mask_register", FormatHex32(mask));
  w.KeyBool("any_active", (status & mask) != 0);

  w.Key("irqs");
  w.StartArray();
  for (u32 i = 0; i < InterruptController::NUM_IRQS; i++)
  {
    w.StartObject();
    w.KeyUint("index", i);
    w.KeyString("name", irq_names[i]);
    w.KeyBool("status", (status & (1u << i)) != 0);
    w.KeyBool("masked", (mask & (1u << i)) != 0);
    w.KeyBool("active", ((status & mask) & (1u << i)) != 0);
    w.EndObject();
  }
  w.EndArray();

  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleGetSpuVoiceState(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  // Optional voice filter
  s32 voice_filter = -1;
  if (args.contains("voice") && args["voice"].is_number_unsigned())
    voice_filter = static_cast<s32>(args["voice"].get_uint());

  if (voice_filter >= 24)
    return ToolResult::Error(-32602, "Voice index must be 0-23");

  // Read global SPU status registers via offsets relative to SPU base 0x1F801C00.
  // ENDX at 0x1F801D9C => offset 0x19C
  const u16 endx_lo = SPU::ReadRegister(0x19C);
  const u16 endx_hi = SPU::ReadRegister(0x19E);
  const u32 endx = static_cast<u32>(endx_lo) | (static_cast<u32>(endx_hi) << 16);

  // Noise mode enable at 0x1F801D94 => offset 0x194
  const u16 noise_lo = SPU::ReadRegister(0x194);
  const u16 noise_hi = SPU::ReadRegister(0x196);
  const u32 noise = static_cast<u32>(noise_lo) | (static_cast<u32>(noise_hi) << 16);

  // Reverb enable at 0x1F801D98 => offset 0x198
  const u16 reverb_lo = SPU::ReadRegister(0x198);
  const u16 reverb_hi = SPU::ReadRegister(0x19A);
  const u32 reverb = static_cast<u32>(reverb_lo) | (static_cast<u32>(reverb_hi) << 16);

  const u32 start = (voice_filter >= 0) ? static_cast<u32>(voice_filter) : 0u;
  const u32 end = (voice_filter >= 0) ? static_cast<u32>(voice_filter + 1) : 24u;

  JsonWriter w;
  w.StartObject();

  w.Key("voices");
  w.StartArray();
  for (u32 i = start; i < end; i++)
  {
    // Voice registers start at offset 0x00 from SPU base, each voice is 0x10 bytes.
    const u16 vol_l = SPU::ReadRegister(i * 0x10 + 0x00);
    const u16 vol_r = SPU::ReadRegister(i * 0x10 + 0x02);
    const u16 pitch = SPU::ReadRegister(i * 0x10 + 0x04);
    const u16 start_addr = SPU::ReadRegister(i * 0x10 + 0x06);
    const u16 adsr_lo = SPU::ReadRegister(i * 0x10 + 0x08);
    const u16 adsr_hi = SPU::ReadRegister(i * 0x10 + 0x0A);
    const u16 adsr_vol = SPU::ReadRegister(i * 0x10 + 0x0C);
    const u16 repeat_addr = SPU::ReadRegister(i * 0x10 + 0x0E);

    w.StartObject();
    w.KeyUint("voice", i);
    w.KeyInt("volume_left", static_cast<s16>(vol_l));
    w.KeyInt("volume_right", static_cast<s16>(vol_r));
    w.KeyUint("pitch", pitch);
    w.KeyString("start_address", fmt::format("0x{:04X}", start_addr));
    w.KeyString("adsr_lo", fmt::format("0x{:04X}", adsr_lo));
    w.KeyString("adsr_hi", fmt::format("0x{:04X}", adsr_hi));
    w.KeyUint("adsr_volume", adsr_vol);
    w.KeyString("repeat_address", fmt::format("0x{:04X}", repeat_addr));
    w.KeyBool("endx", (endx & (1u << i)) != 0);
    w.KeyBool("noise_enabled", (noise & (1u << i)) != 0);
    w.KeyBool("reverb_enabled", (reverb & (1u << i)) != 0);
    w.EndObject();
  }
  w.EndArray();

  w.KeyString("endx_register", FormatHex32(endx));
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleGetGpuDrawState([[maybe_unused]] const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  // GPUSTAT contains many draw-related fields.
  const u32 gpustat = g_gpu.ReadRegister(0x04);

  // Decode GPUSTAT draw-related fields.
  const u8 tex_page_x = (gpustat >> 0) & 0xF;
  const u8 tex_page_y = (gpustat >> 4) & 0x1;
  const u8 semi_transparency = (gpustat >> 5) & 0x3;
  const u8 tex_color_mode = (gpustat >> 7) & 0x3;
  const bool dither_enable = (gpustat >> 9) & 0x1;
  const bool draw_to_display = (gpustat >> 10) & 0x1;
  const bool set_mask = (gpustat >> 11) & 0x1;
  const bool check_mask = (gpustat >> 12) & 0x1;
  const bool texture_disable = (gpustat >> 15) & 0x1;

  static constexpr const char* tex_mode_names[] = {"4-bit CLUT", "8-bit CLUT", "15-bit Direct", "Reserved"};
  static constexpr const char* semi_trans_names[] = {
    "B/2+F/2", "B+F", "B-F", "B+F/4"
  };

  JsonWriter w;
  w.StartObject();
  w.KeyString("gpustat", FormatHex32(gpustat));
  w.KeyUint("texture_page_x", tex_page_x * 64);
  w.KeyUint("texture_page_y", tex_page_y * 256);
  w.KeyString("semi_transparency_mode", semi_trans_names[semi_transparency]);
  w.KeyString("texture_color_mode", tex_mode_names[tex_color_mode]);
  w.KeyBool("dither_enable", dither_enable);
  w.KeyBool("draw_to_displayed_field", draw_to_display);
  w.KeyBool("set_mask_while_drawing", set_mask);
  w.KeyBool("check_mask_before_draw", check_mask);
  w.KeyBool("texture_disable", texture_disable);
  w.KeyBool("display_disable", (bool)((gpustat >> 23) & 0x1));
  w.KeyUint("interlace_field", (gpustat >> 13) & 0x1);
  w.KeyBool("vertical_interlace", (bool)((gpustat >> 22) & 0x1));
  w.KeyBool("pal_mode", (bool)((gpustat >> 20) & 0x1));
  w.KeyBool("display_area_24bit", (bool)((gpustat >> 21) & 0x1));
  w.KeyUint("dma_direction", (gpustat >> 29) & 0x3);
  w.KeyBool("gpu_idle", (bool)((gpustat >> 26) & 0x1));
  w.KeyBool("ready_to_receive_dma", (bool)((gpustat >> 28) & 0x1));
  w.KeyBool("ready_to_send_vram", (bool)((gpustat >> 27) & 0x1));
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleGetGpuStats([[maybe_unused]] const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  // GPU statistics are available via the string formatters on the backend.
  // Since s_stats/s_counters are protected, we use GetStatsString/GetMemoryStatsString
  // which are public, but require a GPUBackend pointer (only on video thread).
  // Instead, report what we can from the CPU-side GPU object.
  const auto [display_w, display_h] = g_gpu.GetFullDisplayResolution();
  const float hfreq = g_gpu.ComputeHorizontalFrequency();
  const float vfreq = g_gpu.ComputeVerticalFrequency();
  const float pixel_ar = g_gpu.ComputePixelAspectRatio();

  JsonWriter w;
  w.StartObject();
  w.KeyUint("display_width", display_w);
  w.KeyUint("display_height", display_h);
  w.KeyDouble("horizontal_frequency", hfreq);
  w.KeyDouble("vertical_frequency", vfreq);
  w.KeyDouble("pixel_aspect_ratio", pixel_ar);
  w.KeyUint("resolution_scale", g_gpu_settings.gpu_resolution_scale);
  w.KeyBool("pal_mode", g_gpu.IsInPALMode());
  w.KeyBool("interlaced", g_gpu.IsInterlacedDisplayEnabled());
  w.KeyBool("display_disabled", g_gpu.IsDisplayDisabled());
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleGetCdromExtendedState([[maybe_unused]] const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  const bool has_media = CDROM::HasMedia();
  const u8 status_reg = CDROM::ReadRegister(0);
  const DiscRegion disc_region = CDROM::GetDiscRegion();

  JsonWriter w;
  w.StartObject();
  w.KeyBool("has_media", has_media);
  w.KeyString("status_register", FormatHex32(status_reg));
  w.KeyString("disc_region", Settings::GetDiscRegionName(disc_region));
  w.KeyBool("is_ps1_disc", has_media ? CDROM::IsMediaPS1Disc() : false);
  w.KeyBool("is_audio_cd", has_media ? CDROM::IsMediaAudioCD() : false);
  w.KeyUint("current_sub_image", CDROM::GetCurrentSubImage());

  if (has_media)
  {
    const CDImage* media = CDROM::GetMedia();
    if (media)
    {
      w.KeyUint("current_lba", media->GetPositionOnDisc());
      w.KeyUint("current_track", media->GetTrackNumber());
      w.KeyUint("track_count", media->GetTrackCount());
      w.KeyUint("disc_size_lba", media->GetLBACount());
      w.KeyString("media_path", std::string(CDROM::GetMediaPath()));

      // Sub-Q position
      const CDImage::Position pos = CDImage::Position::FromLBA(media->GetPositionOnDisc());
      w.KeyUint("position_mm", pos.minute);
      w.KeyUint("position_ss", pos.second);
      w.KeyUint("position_ff", pos.frame);
    }
  }

  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleGetMdecState([[maybe_unused]] const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  const u32 status = MDEC::ReadRegister(0x04); // Status register at offset 4
  const bool active = MDEC::IsActive();
  const bool decoding = MDEC::IsDecodingMacroblock();

  JsonWriter w;
  w.StartObject();
  w.KeyString("status_register", FormatHex32(status));
  w.KeyBool("active", active);
  w.KeyBool("decoding_macroblock", decoding);
  w.KeyBool("data_out_fifo_empty", (bool)((status >> 31) & 1));
  w.KeyBool("data_in_fifo_full", (bool)((status >> 30) & 1));
  w.KeyBool("command_busy", (bool)((status >> 29) & 1));
  w.KeyBool("data_in_request", (bool)((status >> 28) & 1));
  w.KeyBool("data_out_request", (bool)((status >> 27) & 1));
  w.KeyUint("current_block", (status >> 16) & 0x7);
  w.KeyUint("remaining_parameters", status & 0xFFFF);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleGetMemoryMap([[maybe_unused]] const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  static constexpr const char* region_names[] = {
    "RAM", "RAM Mirror 1", "RAM Mirror 2", "RAM Mirror 3", "EXP1", "Scratchpad", "BIOS"
  };
  static_assert(countof(region_names) == static_cast<u32>(Bus::MemoryRegion::Count));

  JsonWriter w;
  w.StartObject();
  w.KeyUint("ram_size", Bus::g_ram_size);

  w.Key("memory_regions");
  w.StartArray();
  for (u32 i = 0; i < static_cast<u32>(Bus::MemoryRegion::Count); i++)
  {
    const auto region = static_cast<Bus::MemoryRegion>(i);
    const u32 start = Bus::GetMemoryRegionStart(region);
    const u32 end = Bus::GetMemoryRegionEnd(region);
    const bool writable = Bus::IsMemoryRegionWritable(region);

    w.StartObject();
    w.KeyString("name", region_names[i]);
    w.KeyString("start", FormatHex32(start));
    w.KeyString("end", FormatHex32(end));
    w.KeyUint("size", end - start);
    w.KeyBool("writable", writable);
    w.EndObject();
  }
  w.EndArray();

  // Add hardware register regions
  w.Key("hardware_registers");
  w.StartArray();

  w.StartObject();
  w.KeyString("name", "Memory Control");
  w.KeyString("start", FormatHex32(Bus::MEMCTRL_BASE));
  w.KeyUint("size", Bus::MEMCTRL_SIZE);
  w.EndObject();

  w.StartObject();
  w.KeyString("name", "PAD/SIO");
  w.KeyString("start", FormatHex32(Bus::PAD_BASE));
  w.KeyUint("size", Bus::PAD_SIZE);
  w.EndObject();

  w.StartObject();
  w.KeyString("name", "SIO");
  w.KeyString("start", FormatHex32(Bus::SIO_BASE));
  w.KeyUint("size", Bus::SIO_SIZE);
  w.EndObject();

  w.StartObject();
  w.KeyString("name", "Interrupt Controller");
  w.KeyString("start", FormatHex32(Bus::INTC_BASE));
  w.KeyUint("size", Bus::INTC_SIZE);
  w.EndObject();

  w.StartObject();
  w.KeyString("name", "DMA");
  w.KeyString("start", FormatHex32(Bus::DMA_BASE));
  w.KeyUint("size", Bus::DMA_SIZE);
  w.EndObject();

  w.StartObject();
  w.KeyString("name", "Timers");
  w.KeyString("start", FormatHex32(Bus::TIMERS_BASE));
  w.KeyUint("size", Bus::TIMERS_SIZE);
  w.EndObject();

  w.StartObject();
  w.KeyString("name", "CDROM");
  w.KeyString("start", FormatHex32(Bus::CDROM_BASE));
  w.KeyUint("size", Bus::CDROM_SIZE);
  w.EndObject();

  w.StartObject();
  w.KeyString("name", "GPU");
  w.KeyString("start", FormatHex32(Bus::GPU_BASE));
  w.KeyUint("size", Bus::GPU_SIZE);
  w.EndObject();

  w.StartObject();
  w.KeyString("name", "MDEC");
  w.KeyString("start", FormatHex32(Bus::MDEC_BASE));
  w.KeyUint("size", Bus::MDEC_SIZE);
  w.EndObject();

  w.StartObject();
  w.KeyString("name", "SPU");
  w.KeyString("start", FormatHex32(Bus::SPU_BASE));
  w.KeyUint("size", Bus::SPU_SIZE);
  w.EndObject();

  w.EndArray();

  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleGetTimingEvents([[maybe_unused]] const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  const TimingEvent* event = *TimingEvents::GetHeadEventPtr();
  const GlobalTicks global_ticks = TimingEvents::GetGlobalTickCounter();

  JsonWriter w;
  w.StartObject();
  w.KeyInt("global_tick_counter", static_cast<s64>(global_ticks));

  // Count events for the event_count field
  u32 event_count = 0;
  {
    const TimingEvent* e = event;
    while (e)
    {
      event_count++;
      e = e->next;
    }
  }
  w.KeyUint("event_count", event_count);

  w.Key("events");
  w.StartArray();
  while (event)
  {
    w.StartObject();
    w.KeyString("name", std::string(event->GetName()));
    w.KeyBool("active", event->IsActive());
    w.KeyInt("period", event->GetPeriod());
    w.KeyInt("interval", event->GetInterval());
    w.KeyInt("next_run_time", static_cast<s64>(event->m_next_run_time));
    w.KeyInt("last_run_time", static_cast<s64>(event->m_last_run_time));
    w.EndObject();
    event = event->next;
  }
  w.EndArray();

  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

// ---- Batch 4: System/Game Tool Handlers ----

static ToolResult HandleInjectExecutable(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (!args.contains("path") || !args["path"].is_string())
    return ToolResult::Error(-32602, "Missing 'path' parameter");

  const std::string path = std::string(args["path"].get_string());

  bool set_pc = true;
  if (args.contains("set_pc") && args["set_pc"].is_bool())
    set_pc = args["set_pc"].get_bool();

  // Read the executable file
  Error error;
  auto data = FileSystem::ReadBinaryFile(path.c_str(), &error);
  if (!data.has_value())
    return ToolResult::Error(-2, fmt::format("Failed to read file: {}", error.GetDescription()));

  if (!Bus::InjectExecutable(data->cspan(), set_pc, &error))
    return ToolResult::Error(-2, fmt::format("Failed to inject executable: {}", error.GetDescription()));

  JsonWriter w;
  w.StartObject();
  w.KeyBool("success", true);
  w.KeyString("path", path);
  w.KeyUint("size", data->size());
  w.KeyBool("set_pc", set_pc);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleStartGpuDump(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (g_gpu.GetGPUDump())
    return ToolResult::Error(-2, "GPU dump already in progress");

  u32 num_frames = 1;
  if (args.contains("num_frames") && args["num_frames"].is_number_unsigned())
    num_frames = static_cast<u32>(args["num_frames"].get_uint());

  const char* path = nullptr;
  std::string path_str;
  if (args.contains("path") && args["path"].is_string())
  {
    path_str = std::string(args["path"].get_string());
    path = path_str.c_str();
  }

  if (!System::StartRecordingGPUDump(path, num_frames))
    return ToolResult::Error(-2, "Failed to start GPU dump recording");

  JsonWriter w;
  w.StartObject();
  w.KeyBool("success", true);
  w.KeyUint("num_frames", num_frames);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleStopGpuDump([[maybe_unused]] const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (!g_gpu.GetGPUDump())
    return ToolResult::Error(-2, "No GPU dump in progress");

  System::StopRecordingGPUDump();

  JsonWriter w;
  w.StartObject();
  w.KeyBool("success", true);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleGetAchievementsState([[maybe_unused]] const JsonValue& args)
{
  JsonWriter w;
  w.StartObject();
  w.KeyBool("active", Achievements::IsActive());
  w.KeyBool("logged_in", Achievements::IsLoggedIn());
  w.KeyBool("hardcore_mode", Achievements::IsHardcoreModeActive());
  w.KeyBool("has_active_game", Achievements::HasActiveGame());

  if (Achievements::IsLoggedIn())
    w.KeyString("username", Achievements::GetLoggedInUserName());

  if (Achievements::HasActiveGame())
  {
    w.KeyUint("game_id", Achievements::GetGameID());
    w.KeyBool("has_achievements", Achievements::HasAchievements());
    w.KeyBool("has_leaderboards", Achievements::HasLeaderboards());
    w.KeyBool("has_rich_presence", Achievements::HasRichPresence());

    auto lock = Achievements::GetLock();
    w.KeyString("game_title", Achievements::GetGameTitle());
    if (Achievements::HasRichPresence())
      w.KeyString("rich_presence", Achievements::GetRichPresenceString());
  }

  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

// ---- Batch 5: Advanced State Tool Handlers ----

static ToolResult HandleGetGpuCrtcState([[maybe_unused]] const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  u32 beam_ticks = 0, beam_line = 0;
  g_gpu.GetBeamPosition(&beam_ticks, &beam_line);

  const float hfreq = g_gpu.ComputeHorizontalFrequency();
  const float vfreq = g_gpu.ComputeVerticalFrequency();
  const GSVector2i video_size = g_gpu.GetCRTCVideoSize();
  const GSVector4i active_rect = g_gpu.GetCRTCVideoActiveRect();
  const GSVector4i vram_rect = g_gpu.GetCRTCVRAMSourceRect();
  const auto [full_w, full_h] = g_gpu.GetFullDisplayResolution();

  JsonWriter w;
  w.StartObject();
  w.KeyUint("beam_ticks", beam_ticks);
  w.KeyUint("beam_line", beam_line);
  w.KeyDouble("horizontal_frequency", hfreq);
  w.KeyDouble("vertical_frequency", vfreq);
  w.KeyInt("video_size_x", video_size.x);
  w.KeyInt("video_size_y", video_size.y);

  w.Key("active_rect");
  w.StartObject();
  w.KeyInt("left", active_rect.x);
  w.KeyInt("top", active_rect.y);
  w.KeyInt("right", active_rect.z);
  w.KeyInt("bottom", active_rect.w);
  w.EndObject();

  w.Key("vram_source_rect");
  w.StartObject();
  w.KeyInt("left", vram_rect.x);
  w.KeyInt("top", vram_rect.y);
  w.KeyInt("right", vram_rect.z);
  w.KeyInt("bottom", vram_rect.w);
  w.EndObject();

  w.KeyUint("full_display_width", full_w);
  w.KeyUint("full_display_height", full_h);
  w.KeyUint("active_start_line", g_gpu.GetCRTCActiveStartLine());
  w.KeyUint("active_end_line", g_gpu.GetCRTCActiveEndLine());
  w.KeyBool("pal_mode", g_gpu.IsInPALMode());
  w.KeyBool("interlaced_display", g_gpu.IsInterlacedDisplayEnabled());
  w.KeyBool("display_disabled", g_gpu.IsDisplayDisabled());
  w.KeyDouble("pixel_aspect_ratio", g_gpu.ComputePixelAspectRatio());
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleGetSpuReverbState([[maybe_unused]] const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  // SPU reverb configuration registers (offsets relative to SPU base 0x1F801C00).
  // Reverb output volume: 0x1F801D84 => offset 0x184
  const u16 reverb_out_l = SPU::ReadRegister(0x184);
  const u16 reverb_out_r = SPU::ReadRegister(0x186);

  // Reverb work area start: 0x1F801DA2 => offset 0x1A2
  const u16 reverb_base = SPU::ReadRegister(0x1A2);

  // SPU control register for reverb enable bit: 0x1F801DAA => offset 0x1AA
  const u16 spucnt = SPU::ReadRegister(0x1AA);
  const bool reverb_master_enable = (spucnt >> 7) & 1;

  // Reverb registers at 0x1F801DC0..0x1F801DFF => offsets 0x1C0..0x1FF
  static constexpr const char* reverb_reg_names[] = {
    "dAPF1", "dAPF2", "vIIR", "vCOMB1", "vCOMB2", "vCOMB3", "vCOMB4", "vWALL",
    "vAPF1", "vAPF2", "mLSAME", "mRSAME", "mLCOMB1", "mRCOMB1", "mLCOMB2", "mRCOMB2",
    "dLSAME", "dRSAME", "mLDIFF", "mRDIFF", "mLCOMB3", "mRCOMB3", "mLCOMB4", "mRCOMB4",
    "dLDIFF", "dRDIFF", "mLAPF1", "mRAPF1", "mLAPF2", "mRAPF2", "vLIN", "vRIN"
  };

  JsonWriter w;
  w.StartObject();
  w.KeyBool("reverb_master_enable", reverb_master_enable);
  w.KeyInt("reverb_output_volume_left", static_cast<s16>(reverb_out_l));
  w.KeyInt("reverb_output_volume_right", static_cast<s16>(reverb_out_r));
  w.KeyString("reverb_work_area_start", fmt::format("0x{:04X}", reverb_base));

  w.Key("reverb_registers");
  w.StartObject();
  for (u32 i = 0; i < 32; i++)
  {
    const u16 val = SPU::ReadRegister(0x1C0 + i * 2);
    w.KeyString(reverb_reg_names[i], fmt::format("0x{:04X}", val));
  }
  w.EndObject();

  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleGetCpuIcacheState([[maybe_unused]] const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  const auto& state = CPU::g_state;

  // Report cache control register state
  const bool icache_enabled = state.cache_control.icache_enable;
  const bool dcache_enabled = state.cache_control.dcache_enable;
  const bool isolate_cache = state.cop0_regs.sr.Isc;
  const bool tag_test = state.cache_control.tag_test_mode;

  // Optionally filter to specific address range
  u32 filter_addr = 0;
  bool has_filter = false;
  if (args.contains("address"))
  {
    const auto addr = ParseAddress(args["address"]);
    if (addr.has_value())
    {
      filter_addr = addr.value();
      has_filter = true;
    }
  }

  u32 valid_lines = 0;
  for (u32 line = 0; line < CPU::ICACHE_LINES; line++)
  {
    const u32 tag = state.icache_tags[line];
    const bool valid = (tag & CPU::ICACHE_INVALID_BITS) == 0;
    if (valid)
      valid_lines++;
  }

  JsonWriter w;
  w.StartObject();
  w.KeyString("cache_control", FormatHex32(state.cache_control.bits));
  w.KeyBool("icache_enabled", icache_enabled);
  w.KeyBool("dcache_enabled", dcache_enabled);
  w.KeyBool("isolate_cache", isolate_cache);
  w.KeyBool("tag_test_mode", tag_test);
  w.KeyUint("total_lines", CPU::ICACHE_LINES);
  w.KeyUint("valid_lines", valid_lines);
  w.KeyUint("line_size", CPU::ICACHE_LINE_SIZE);
  w.KeyUint("total_size", CPU::ICACHE_SIZE);

  w.Key("cache_lines");
  w.StartArray();
  for (u32 line = 0; line < CPU::ICACHE_LINES; line++)
  {
    const u32 tag = state.icache_tags[line];
    const u32 tag_addr = tag & CPU::ICACHE_TAG_ADDRESS_MASK;
    const bool valid = (tag & CPU::ICACHE_INVALID_BITS) == 0;

    if (has_filter)
    {
      // Only show lines matching the filter address region
      const u32 line_start = tag_addr;
      const u32 line_end = tag_addr + CPU::ICACHE_LINE_SIZE;
      if (filter_addr < line_start || filter_addr >= line_end)
        continue;
    }

    // Only include first 32 lines unless filtering (to avoid huge output)
    if (!has_filter && line >= 32)
      continue;

    w.StartObject();
    w.KeyUint("line", line);
    w.KeyString("tag", FormatHex32(tag));
    w.KeyString("tag_address", FormatHex32(tag_addr));
    w.KeyBool("valid", valid);

    // Include cached words
    w.Key("words");
    w.StartArray();
    for (u32 wi = 0; wi < CPU::ICACHE_WORDS_PER_LINE; wi++)
      w.String(FormatHex32(state.icache_data[line * CPU::ICACHE_WORDS_PER_LINE + wi]));
    w.EndArray();

    w.EndObject();
  }
  w.EndArray();

  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleGetPgxpState([[maybe_unused]] const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  JsonWriter w;
  w.StartObject();
  w.KeyBool("pgxp_enabled", (bool)g_settings.gpu_pgxp_enable);
  w.KeyBool("pgxp_culling", (bool)g_settings.gpu_pgxp_culling);
  w.KeyBool("pgxp_texture_correction", (bool)g_settings.gpu_pgxp_texture_correction);
  w.KeyBool("pgxp_color_correction", (bool)g_settings.gpu_pgxp_color_correction);
  w.KeyBool("pgxp_vertex_cache", (bool)g_settings.gpu_pgxp_vertex_cache);
  w.KeyBool("pgxp_cpu_mode", (bool)g_settings.gpu_pgxp_cpu);
  w.KeyBool("pgxp_depth_buffer", (bool)g_settings.gpu_pgxp_depth_buffer);
  w.KeyBool("pgxp_disable_2d", (bool)g_settings.gpu_pgxp_disable_2d);
  w.KeyDouble("pgxp_tolerance", g_settings.gpu_pgxp_tolerance);
  w.KeyDouble("pgxp_depth_clear_threshold", g_settings.gpu_pgxp_depth_clear_threshold);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleGetEnhancedStatus([[maybe_unused]] const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  JsonWriter w;
  w.StartObject();

  // Basic system info
  w.KeyString("state", System::IsPaused() ? "paused" : "running");
  w.KeyString("game_serial", System::GetGameSerial());
  w.KeyString("game_title", System::GetGameTitle());
  w.KeyString("game_path", System::GetGamePath());
  w.KeyString("region", Settings::GetConsoleRegionName(System::GetRegion()));
  w.KeyUint("frame_number", System::GetFrameNumber());
  w.KeyUint("internal_frame_number", System::GetInternalFrameNumber());
  w.KeyInt("global_tick_counter", static_cast<s64>(System::GetGlobalTickCounter()));
  w.KeyDouble("session_played_time_seconds", System::GetSessionPlayedTime());
  w.KeyUint("boot_mode", static_cast<u8>(System::GetBootMode()));

  // Taints
  w.Key("taints");
  w.StartArray();
  for (u32 i = 0; i < static_cast<u32>(System::Taint::MaxCount); i++)
  {
    const auto taint = static_cast<System::Taint>(i);
    if (System::HasTaint(taint))
    {
      w.StartObject();
      w.KeyString("name", System::GetTaintName(taint));
      w.KeyString("display_name", std::string(System::GetTaintDisplayName(taint)));
      w.EndObject();
    }
  }
  w.EndArray();

  // Speed info
  w.KeyDouble("target_speed", System::GetTargetSpeed());
  w.KeyDouble("video_frame_rate", System::GetVideoFrameRate());
  w.KeyBool("fast_forward", System::IsFastForwardEnabled());
  w.KeyBool("turbo", System::IsTurboEnabled());
  w.KeyBool("rewinding", System::IsRewinding());

  // Latency stats
  SmallString latency_str;
  System::FormatLatencyStats(latency_str);
  w.KeyString("latency_stats", std::string(latency_str.c_str()));

  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

// =============================================================================
// BATCH 1: Game List (6 tools)
// =============================================================================

static void WriteGameListEntryJson(JsonWriter& w, const GameList::Entry& e)
{
  w.StartObject();
  w.KeyString("title", e.title);
  w.KeyString("serial", e.serial);
  w.KeyString("path", e.path);
  w.KeyString("type", GameList::GetEntryTypeName(e.type));
  w.KeyString("region", Settings::GetDiscRegionName(e.region));
  w.KeyInt("file_size", e.file_size);
  w.KeyInt("last_modified_time", static_cast<s64>(e.last_modified_time));
  w.KeyInt("last_played_time", static_cast<s64>(e.last_played_time));
  w.KeyInt("total_played_time", static_cast<s64>(e.total_played_time));

  if (e.total_played_time > 0)
    w.KeyString("total_played_time_formatted", GameList::FormatTimespan(e.total_played_time, true));
  else
    w.KeyString("total_played_time_formatted", "Never played");

  if (e.last_played_time > 0)
    w.KeyString("last_played_time_formatted", GameList::FormatTimestamp(e.last_played_time));
  else
    w.KeyString("last_played_time_formatted", "Never");

  if (e.dbentry)
  {
    w.Key("database");
    w.StartObject();
    if (!e.dbentry->developer.empty())
      w.KeyString("developer", e.dbentry->developer);
    if (!e.dbentry->publisher.empty())
      w.KeyString("publisher", e.dbentry->publisher);
    if (e.dbentry->release_date != 0)
      w.KeyString("release_date", e.GetReleaseDateString());
    w.KeyUint("min_players", e.dbentry->min_players);
    w.KeyUint("max_players", e.dbentry->max_players);
    w.EndObject();
  }

  w.EndObject();
}

// Tool 1: list_games
static ToolResult HandleListGames(const JsonValue& args)
{
  const std::string_view filter =
    (args.contains("filter") && args["filter"].is_string()) ? args["filter"].get_string() : std::string_view();
  const std::string_view sort_by =
    (args.contains("sort_by") && args["sort_by"].is_string()) ? args["sort_by"].get_string() : std::string_view("title");
  const s64 max_results =
    (args.contains("max_results") && args["max_results"].is_number_integer()) ? args["max_results"].get_int() : 100;

  auto lock = GameList::GetLock();
  const std::span<const GameList::Entry> entries = GameList::GetEntries();

  // Collect matching entries (as pointers to avoid copies).
  std::vector<const GameList::Entry*> matched;
  matched.reserve(entries.size());
  for (const auto& e : entries)
  {
    if (!filter.empty())
    {
      if (!StringUtil::ContainsNoCase(e.title, filter) &&
          !StringUtil::ContainsNoCase(e.serial, filter) &&
          !StringUtil::ContainsNoCase(e.path, filter))
      {
        continue;
      }
    }
    matched.push_back(&e);
  }

  // Sort by the requested field.
  if (sort_by == "serial")
  {
    std::sort(matched.begin(), matched.end(),
              [](const GameList::Entry* a, const GameList::Entry* b) { return a->serial < b->serial; });
  }
  else if (sort_by == "region")
  {
    std::sort(matched.begin(), matched.end(), [](const GameList::Entry* a, const GameList::Entry* b) {
      return Settings::GetDiscRegionName(a->region) < Settings::GetDiscRegionName(b->region);
    });
  }
  else if (sort_by == "size")
  {
    std::sort(matched.begin(), matched.end(),
              [](const GameList::Entry* a, const GameList::Entry* b) { return a->file_size > b->file_size; });
  }
  else if (sort_by == "last_played")
  {
    std::sort(matched.begin(), matched.end(), [](const GameList::Entry* a, const GameList::Entry* b) {
      return a->last_played_time > b->last_played_time;
    });
  }
  else // default: title
  {
    std::sort(matched.begin(), matched.end(),
              [](const GameList::Entry* a, const GameList::Entry* b) { return a->title < b->title; });
  }

  // Clamp to max_results.
  const size_t count = (max_results > 0) ? std::min(matched.size(), static_cast<size_t>(max_results)) : matched.size();

  JsonWriter w;
  w.StartObject();
  w.KeyUint("total_entries", static_cast<u64>(entries.size()));
  w.KeyUint("matched_entries", static_cast<u64>(matched.size()));
  w.KeyUint("returned_entries", static_cast<u64>(count));

  w.Key("games");
  w.StartArray();
  for (size_t i = 0; i < count; i++)
    WriteGameListEntryJson(w, *matched[i]);
  w.EndArray();

  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

// Tool 2: get_game_info
static ToolResult HandleGetGameInfo(const JsonValue& args)
{
  const bool has_serial = args.contains("serial") && args["serial"].is_string();
  const bool has_path = args.contains("path") && args["path"].is_string();

  if (!has_serial && !has_path)
    return ToolResult::Error(-32602, "One of 'serial' or 'path' is required");

  auto lock = GameList::GetLock();
  const GameList::Entry* entry = nullptr;

  if (has_serial)
    entry = GameList::GetEntryBySerial(args["serial"].get_string());
  if (!entry && has_path)
    entry = GameList::GetEntryForPath(args["path"].get_string());

  if (!entry)
    return ToolResult::Error(-1, "Game not found in game list");

  JsonWriter w;
  w.StartObject();
  w.Key("game");
  WriteGameListEntryJson(w, *entry);

  // Also include cover image path if available.
  const std::string cover_path = GameList::GetCoverImagePathForEntry(entry);
  if (!cover_path.empty())
    w.KeyString("cover_image_path", cover_path);
  else
    w.KeyNull("cover_image_path");

  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

// Tool 4: refresh_game_list
static ToolResult HandleRefreshGameList(const JsonValue& args)
{
  const bool invalidate_cache =
    (args.contains("invalidate_cache") && args["invalidate_cache"].is_bool()) ? args["invalidate_cache"].get_bool()
                                                                              : false;

  GameList::Refresh(invalidate_cache);

  auto lock = GameList::GetLock();
  const size_t count = GameList::GetEntries().size();

  JsonWriter w;
  w.StartObject();
  w.KeyString("status", "refreshed");
  w.KeyBool("invalidate_cache", invalidate_cache);
  w.KeyUint("total_entries", static_cast<u64>(count));
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

// =============================================================================
// BATCH 2: BIOS (2 tools)
// =============================================================================

// Tool 7: list_bios
static ToolResult HandleListBios([[maybe_unused]] const JsonValue& args)
{
  const auto images = BIOS::FindBIOSImagesInDirectory(EmuFolders::Bios.c_str());

  JsonWriter w;
  w.StartObject();
  w.KeyString("bios_directory", EmuFolders::Bios);
  w.KeyUint("count", static_cast<u64>(images.size()));

  w.Key("images");
  w.StartArray();
  for (const auto& [filename, info] : images)
  {
    w.StartObject();
    w.KeyString("filename", filename);
    if (info)
    {
      w.KeyString("description", info->description ? info->description : "Unknown");
      w.KeyString("region", Settings::GetConsoleRegionName(info->region));
      w.KeyString("hash", BIOS::ImageInfo::GetHashString(info->hash));
      w.KeyBool("supports_fast_boot", info->SupportsFastBoot());
      w.KeyBool("region_check", info->region_check);
      w.KeyUint("priority", info->priority);
    }
    else
    {
      w.KeyString("description", "Unknown/unrecognized BIOS");
      w.KeyNull("region");
      w.KeyNull("hash");
      w.KeyBool("supports_fast_boot", false);
      w.KeyBool("region_check", false);
      w.KeyNull("priority");
    }
    w.EndObject();
  }
  w.EndArray();

  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

// Tool 8: get_bios_info
static ToolResult HandleGetBiosInfo(const JsonValue& args)
{
  if (!args.contains("filename") || !args["filename"].is_string())
    return ToolResult::Error(-32602, "Missing required 'filename' parameter");

  const std::string_view target_filename = args["filename"].get_string();
  const auto images = BIOS::FindBIOSImagesInDirectory(EmuFolders::Bios.c_str());

  for (const auto& [filename, info] : images)
  {
    if (filename == target_filename)
    {
      JsonWriter w;
      w.StartObject();
      w.KeyString("filename", filename);
      w.KeyString("bios_directory", EmuFolders::Bios);

      if (info)
      {
        w.KeyString("description", info->description ? info->description : "Unknown");
        w.KeyString("region", Settings::GetConsoleRegionName(info->region));
        w.KeyString("hash", BIOS::ImageInfo::GetHashString(info->hash));
        w.KeyBool("supports_fast_boot", info->SupportsFastBoot());
        w.KeyBool("region_check", info->region_check);
        w.KeyUint("priority", info->priority);

        switch (info->fastboot_patch)
        {
          case BIOS::ImageInfo::FastBootPatch::Unsupported:
            w.KeyString("fastboot_patch_type", "unsupported");
            break;
          case BIOS::ImageInfo::FastBootPatch::Type1:
            w.KeyString("fastboot_patch_type", "type1");
            break;
          case BIOS::ImageInfo::FastBootPatch::Type2:
            w.KeyString("fastboot_patch_type", "type2");
            break;
        }
      }
      else
      {
        w.KeyString("description", "Unknown/unrecognized BIOS");
        w.KeyNull("region");
        w.KeyNull("hash");
        w.KeyBool("supports_fast_boot", false);
        w.KeyBool("region_check", false);
        w.KeyNull("priority");
        w.KeyString("fastboot_patch_type", "unknown");
      }

      w.EndObject();
      return ToolResult{w.TakeOutput()};
    }
  }

  return ToolResult::Error(-1, fmt::format("BIOS file '{}' not found in directory '{}'", target_filename,
                                           EmuFolders::Bios));
}

// =============================================================================
// BATCH 3: Save State Extended (4 tools)
// =============================================================================

// Tool 9: get_save_state_info
static ToolResult HandleGetSaveStateInfo(const JsonValue& args)
{
  if (!args.contains("slot") || !args["slot"].is_number_integer())
    return ToolResult::Error(-32602, "Missing or invalid 'slot' parameter (integer required)");

  const s32 slot = static_cast<s32>(args["slot"].get_int());
  const bool global =
    (args.contains("global") && args["global"].is_bool()) ? args["global"].get_bool() : false;

  std::string path;
  if (global)
  {
    path = System::GetGlobalSaveStatePath(slot);
  }
  else
  {
    // Need a serial for game-specific save states.
    std::string serial;
    if (args.contains("serial") && args["serial"].is_string())
      serial = std::string(args["serial"].get_string());
    else if (System::IsValid())
      serial = System::GetGameSerial();

    if (serial.empty())
      return ToolResult::Error(-1, "No serial provided and no game is currently running");

    path = System::GetGameSaveStatePath(serial, slot);
  }

  const std::optional<ExtendedSaveStateInfo> info = System::GetExtendedSaveStateInfo(path.c_str());
  if (!info.has_value())
    return ToolResult::Error(-2, fmt::format("Save state not found or invalid at path: {}", path));

  JsonWriter w;
  w.StartObject();
  w.KeyString("path", path);
  w.KeyInt("slot", slot);
  w.KeyBool("global", global);
  w.KeyString("title", info->title);
  w.KeyString("serial", info->serial);
  w.KeyString("media_path", info->media_path);
  w.KeyInt("timestamp", static_cast<s64>(info->timestamp));
  w.KeyString("timestamp_formatted", GameList::FormatTimestamp(info->timestamp));
  w.KeyBool("has_screenshot", info->screenshot.IsValid());

  if (info->screenshot.IsValid())
  {
    w.KeyUint("screenshot_width", info->screenshot.GetWidth());
    w.KeyUint("screenshot_height", info->screenshot.GetHeight());
  }

  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

// Tool 10: delete_save_states
static ToolResult HandleDeleteSaveStates(const JsonValue& args)
{
  if (!args.contains("serial") || !args["serial"].is_string())
    return ToolResult::Error(-32602, "Missing required 'serial' parameter");

  const std::string serial = std::string(args["serial"].get_string());
  if (serial.empty())
    return ToolResult::Error(-32602, "'serial' must be a non-empty string");

  const bool include_resume =
    (args.contains("include_resume") && args["include_resume"].is_bool()) ? args["include_resume"].get_bool() : false;

  System::DeleteSaveStates(serial, include_resume);

  JsonWriter w;
  w.StartObject();
  w.KeyString("status", "deleted");
  w.KeyString("serial", serial);
  w.KeyBool("include_resume", include_resume);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

// Tool 11: undo_load_state
static ToolResult HandleUndoLoadState([[maybe_unused]] const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (!System::CanUndoLoadState())
    return ToolResult::Error(-2, "No undo state available. A state must have been loaded first.");

  System::UndoLoadState();

  JsonWriter w;
  w.StartObject();
  w.KeyString("status", "undone");
  w.KeyString("message", "Previous state restored successfully");
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

// Tool 12: get_save_state_screenshot
static ToolResult HandleGetSaveStateScreenshot(const JsonValue& args)
{
  if (!args.contains("slot") || !args["slot"].is_number_integer())
    return ToolResult::Error(-32602, "Missing or invalid 'slot' parameter (integer required)");

  const s32 slot = static_cast<s32>(args["slot"].get_int());
  const bool global =
    (args.contains("global") && args["global"].is_bool()) ? args["global"].get_bool() : false;

  std::string path;
  if (global)
  {
    path = System::GetGlobalSaveStatePath(slot);
  }
  else
  {
    std::string serial;
    if (args.contains("serial") && args["serial"].is_string())
      serial = std::string(args["serial"].get_string());
    else if (System::IsValid())
      serial = System::GetGameSerial();

    if (serial.empty())
      return ToolResult::Error(-1, "No serial provided and no game is currently running");

    path = System::GetGameSaveStatePath(serial, slot);
  }

  const std::optional<ExtendedSaveStateInfo> info = System::GetExtendedSaveStateInfo(path.c_str());
  if (!info.has_value())
    return ToolResult::Error(-2, fmt::format("Save state not found or invalid at path: {}", path));

  JsonWriter w;
  w.StartObject();
  w.KeyString("path", path);
  w.KeyInt("slot", slot);
  w.KeyBool("global", global);
  w.KeyString("title", info->title);
  w.KeyString("serial", info->serial);

  if (info->screenshot.IsValid())
  {
    w.KeyBool("has_screenshot", true);
    w.KeyUint("width", info->screenshot.GetWidth());
    w.KeyUint("height", info->screenshot.GetHeight());

    const auto png_data = info->screenshot.SaveToBuffer("screenshot.png");
    if (png_data.has_value())
    {
      const std::string temp_path = GetMCPTempFilePath("screenshot", "png");
      if (FileSystem::WriteBinaryFile(temp_path.c_str(), png_data->data(), png_data->size()))
        w.KeyString("screenshot_path", temp_path);
      else
        w.KeyNull("screenshot_path");
    }
    else
    {
      w.KeyNull("screenshot_path");
    }
  }
  else
  {
    w.KeyBool("has_screenshot", false);
  }

  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

// ============================================================================
// BATCH 4: Cheats Advanced (5 tools)
// ============================================================================

static ToolResult HandleGetCheatDetails(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (!args.contains("name") || !args["name"].is_string())
    return ToolResult::Error(-32602, "Missing 'name' parameter");

  const std::string name = std::string(args["name"].get_string());
  const std::string serial = System::GetGameSerial();
  const GameHash hash = System::GetGameHash();

  const Cheats::CodeInfoList codes = Cheats::GetCodeInfoList(serial, hash, true, true, true);
  const Cheats::CodeInfo* code = Cheats::FindCodeInInfoList(codes, name);
  if (!code)
  {
    // Also search patches.
    const Cheats::CodeInfoList patches = Cheats::GetCodeInfoList(serial, hash, false, true, true);
    code = Cheats::FindCodeInInfoList(patches, name);
    if (!code)
      return ToolResult::Error(-2, fmt::format("Cheat not found: {}", name));
  }

  JsonWriter w;
  w.StartObject();
  w.KeyString("name", code->name);
  w.KeyString("author", code->author);
  w.KeyString("description", code->description);
  w.KeyString("body", code->body);
  w.KeyString("type", Cheats::GetTypeName(code->type));
  w.KeyString("activation", Cheats::GetActivationName(code->activation));
  w.KeyBool("from_database", code->from_database);
  w.KeyBool("disallow_for_achievements", code->disallow_for_achievements);

  if (!code->options.empty())
  {
    w.Key("options");
    w.StartArray();
    for (const auto& [opt_name, opt_value] : code->options)
    {
      w.StartObject();
      w.KeyString("name", opt_name);
      w.KeyUint("value", opt_value);
      w.EndObject();
    }
    w.EndArray();
  }

  if (code->HasOptionRange())
  {
    w.KeyUint("option_range_start", code->option_range_start);
    w.KeyUint("option_range_end", code->option_range_end);
  }

  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleCreateCheat(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (!args.contains("name") || !args["name"].is_string())
    return ToolResult::Error(-32602, "Missing 'name' parameter");
  if (!args.contains("body") || !args["body"].is_string())
    return ToolResult::Error(-32602, "Missing 'body' parameter");

  const std::string name = std::string(args["name"].get_string());
  const std::string body = std::string(args["body"].get_string());

  Cheats::CodeType type = Cheats::CodeType::Gameshark;
  if (args.contains("type") && args["type"].is_string())
  {
    const std::optional<Cheats::CodeType> parsed = Cheats::ParseTypeName(args["type"].get_string());
    if (!parsed.has_value())
      return ToolResult::Error(-32602, fmt::format("Unknown cheat type: {}", args["type"].get_string()));
    type = parsed.value();
  }

  Cheats::CodeActivation activation = Cheats::CodeActivation::Manual;
  if (args.contains("activation") && args["activation"].is_string())
  {
    const std::string act_str = std::string(args["activation"].get_string());
    if (act_str == "manual")
      activation = Cheats::CodeActivation::Manual;
    else if (act_str == "end_frame" || act_str == "endframe")
      activation = Cheats::CodeActivation::EndFrame;
    else
    {
      const std::optional<Cheats::CodeActivation> parsed = Cheats::ParseActivationName(act_str);
      if (!parsed.has_value())
        return ToolResult::Error(-32602, fmt::format("Unknown activation type: {}", act_str));
      activation = parsed.value();
    }
  }

  // Validate the code body before saving.
  Error error;
  if (!Cheats::ValidateCodeBody(name, type, activation, body, &error))
    return ToolResult::Error(-3, fmt::format("Invalid cheat body: {}", error.GetDescription()));

  Cheats::CodeInfo code;
  code.name = name;
  code.body = body;
  code.type = type;
  code.activation = activation;

  const std::string serial = System::GetGameSerial();
  const GameHash hash = System::GetGameHash();
  const std::string cht_path = Cheats::GetChtFilename(serial, hash, true);

  if (!Cheats::UpdateCodeInFile(cht_path.c_str(), name, &code, &error))
    return ToolResult::Error(-2, fmt::format("Failed to save cheat: {}", error.GetDescription()));

  JsonWriter w;
  w.StartObject();
  w.KeyString("status", "created");
  w.KeyString("name", name);
  w.KeyString("path", cht_path);
  w.KeyString("type", Cheats::GetTypeName(type));
  w.KeyString("activation", Cheats::GetActivationName(activation));
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleImportCheats(const JsonValue& args)
{
  if (!args.contains("content") || !args["content"].is_string())
    return ToolResult::Error(-32602, "Missing 'content' parameter");
  if (!args.contains("format") || !args["format"].is_string())
    return ToolResult::Error(-32602, "Missing 'format' parameter");

  const std::string content = std::string(args["content"].get_string());
  const std::string format_str = std::string(args["format"].get_string());

  Cheats::FileFormat file_format = Cheats::FileFormat::Unknown;
  if (format_str == "duckstation")
    file_format = Cheats::FileFormat::DuckStation;
  else if (format_str == "pcsx")
    file_format = Cheats::FileFormat::PCSX;
  else if (format_str == "libretro")
    file_format = Cheats::FileFormat::Libretro;
  else if (format_str == "epsxe")
    file_format = Cheats::FileFormat::EPSXe;
  else
    return ToolResult::Error(-32602, fmt::format("Unknown cheat format: {}", format_str));

  Cheats::CodeInfoList imported;
  Error error;
  if (!Cheats::ImportCodesFromString(&imported, content, file_format, false, &error))
    return ToolResult::Error(-2, fmt::format("Failed to import cheats: {}", error.GetDescription()));

  JsonWriter w;
  w.StartObject();
  w.KeyString("status", "imported");
  w.KeyUint("count", static_cast<u32>(imported.size()));

  w.Key("codes");
  w.StartArray();
  for (const auto& code : imported)
  {
    w.StartObject();
    w.KeyString("name", code.name);
    w.KeyString("type", Cheats::GetTypeName(code.type));
    w.KeyString("activation", Cheats::GetActivationName(code.activation));
    w.EndObject();
  }
  w.EndArray();

  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleExportCheats(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (!args.contains("path") || !args["path"].is_string())
    return ToolResult::Error(-32602, "Missing 'path' parameter");

  const std::string path = std::string(args["path"].get_string());
  const std::string serial = System::GetGameSerial();
  const GameHash hash = System::GetGameHash();

  const Cheats::CodeInfoList codes = Cheats::GetCodeInfoList(serial, hash, true, true, true);

  Error error;
  if (!Cheats::ExportCodesToFile(path, codes, &error))
    return ToolResult::Error(-2, fmt::format("Failed to export cheats: {}", error.GetDescription()));

  JsonWriter w;
  w.StartObject();
  w.KeyString("status", "exported");
  w.KeyString("path", path);
  w.KeyUint("count", static_cast<u32>(codes.size()));
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleValidateCheat(const JsonValue& args)
{
  if (!args.contains("body") || !args["body"].is_string())
    return ToolResult::Error(-32602, "Missing 'body' parameter");

  const std::string body = std::string(args["body"].get_string());

  // Parse type and activation if provided, otherwise default to Gameshark/Manual.
  Cheats::CodeType type = Cheats::CodeType::Gameshark;
  if (args.contains("type") && args["type"].is_string())
  {
    const std::optional<Cheats::CodeType> parsed = Cheats::ParseTypeName(args["type"].get_string());
    if (parsed.has_value())
      type = parsed.value();
  }

  Cheats::CodeActivation activation = Cheats::CodeActivation::Manual;
  if (args.contains("activation") && args["activation"].is_string())
  {
    const std::optional<Cheats::CodeActivation> parsed = Cheats::ParseActivationName(args["activation"].get_string());
    if (parsed.has_value())
      activation = parsed.value();
  }

  Error error;
  const bool valid = Cheats::ValidateCodeBody("_validate", type, activation, body, &error);

  JsonWriter w;
  w.StartObject();
  w.KeyBool("valid", valid);
  if (!valid)
    w.KeyString("error", error.GetDescription());
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

// ============================================================================
// BATCH 5: Memory Card Advanced (5 tools)
// ============================================================================

static ToolResult HandleGetMemoryCardInfo(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (!args.contains("slot") || !args["slot"].is_number())
    return ToolResult::Error(-32602, "Missing 'slot' parameter");

  const u32 slot = static_cast<u32>(args["slot"].get_uint());
  if (slot > 1)
    return ToolResult::Error(-32602, "Slot must be 0 or 1");

  const MemoryCard* mc = Pad::GetMemoryCard(slot);
  if (!mc)
    return ToolResult::Error(-2, fmt::format("No memory card in slot {}", slot));

  const MemoryCardImage::DataArray& data = mc->GetData();
  const bool valid = MemoryCardImage::IsValid(data);

  JsonWriter w;
  w.StartObject();
  w.KeyUint("slot", slot);
  w.KeyString("path", mc->GetPath());
  w.KeyBool("valid", valid);

  if (valid)
  {
    const u32 free_blocks = MemoryCardImage::GetFreeBlockCount(data);
    const auto files = MemoryCardImage::EnumerateFiles(data, false);

    w.KeyUint("free_blocks", free_blocks);
    w.KeyUint("total_blocks", MemoryCardImage::NUM_BLOCKS);
    w.KeyUint("file_count", static_cast<u32>(files.size()));

    w.Key("files");
    w.StartArray();
    for (const auto& fi : files)
    {
      w.StartObject();
      w.KeyString("filename", fi.filename);
      w.KeyString("title", fi.title);
      w.KeyUint("size", fi.size);
      w.KeyUint("num_blocks", fi.num_blocks);
      w.KeyBool("deleted", fi.deleted);
      w.EndObject();
    }
    w.EndArray();
  }

  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleReadMemoryCardFile(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (!args.contains("slot") || !args["slot"].is_number())
    return ToolResult::Error(-32602, "Missing 'slot' parameter");
  if (!args.contains("filename") || !args["filename"].is_string())
    return ToolResult::Error(-32602, "Missing 'filename' parameter");

  const u32 slot = static_cast<u32>(args["slot"].get_uint());
  if (slot > 1)
    return ToolResult::Error(-32602, "Slot must be 0 or 1");

  const MemoryCard* mc = Pad::GetMemoryCard(slot);
  if (!mc)
    return ToolResult::Error(-2, fmt::format("No memory card in slot {}", slot));

  const MemoryCardImage::DataArray& data = mc->GetData();
  if (!MemoryCardImage::IsValid(data))
    return ToolResult::Error(-2, "Memory card data is not valid");

  const std::string filename = std::string(args["filename"].get_string());
  const auto files = MemoryCardImage::EnumerateFiles(data, false);

  const MemoryCardImage::FileInfo* target = nullptr;
  for (const auto& fi : files)
  {
    if (fi.filename == filename)
    {
      target = &fi;
      break;
    }
  }

  if (!target)
    return ToolResult::Error(-2, fmt::format("File not found on memory card: {}", filename));

  std::vector<u8> buffer;
  Error error;
  if (!MemoryCardImage::ReadFile(data, *target, &buffer, &error))
    return ToolResult::Error(-2, fmt::format("Failed to read file: {}", error.GetDescription()));

  const std::string output_path = GetMCPTempFilePath("mcfile", "bin");
  if (!FileSystem::WriteBinaryFile(output_path.c_str(), buffer.data(), buffer.size(), &error))
    return ToolResult::Error(-2, fmt::format("Failed to write temp file: {}", error.GetDescription()));

  JsonWriter w;
  w.StartObject();
  w.KeyUint("slot", slot);
  w.KeyString("filename", target->filename);
  w.KeyString("title", target->title);
  w.KeyUint("size", target->size);
  w.KeyUint("num_blocks", target->num_blocks);
  w.KeyString("output_path", output_path);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleDeleteMemoryCardFile(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (!args.contains("slot") || !args["slot"].is_number())
    return ToolResult::Error(-32602, "Missing 'slot' parameter");
  if (!args.contains("filename") || !args["filename"].is_string())
    return ToolResult::Error(-32602, "Missing 'filename' parameter");

  const u32 slot = static_cast<u32>(args["slot"].get_uint());
  if (slot > 1)
    return ToolResult::Error(-32602, "Slot must be 0 or 1");

  MemoryCard* mc = Pad::GetMemoryCard(slot);
  if (!mc)
    return ToolResult::Error(-2, fmt::format("No memory card in slot {}", slot));

  MemoryCardImage::DataArray& data = mc->GetData();
  if (!MemoryCardImage::IsValid(data))
    return ToolResult::Error(-2, "Memory card data is not valid");

  const std::string filename = std::string(args["filename"].get_string());
  const auto files = MemoryCardImage::EnumerateFiles(data, false);

  const MemoryCardImage::FileInfo* target = nullptr;
  for (const auto& fi : files)
  {
    if (fi.filename == filename)
    {
      target = &fi;
      break;
    }
  }

  if (!target)
    return ToolResult::Error(-2, fmt::format("File not found on memory card: {}", filename));

  if (!MemoryCardImage::DeleteFile(&data, *target, true))
    return ToolResult::Error(-2, fmt::format("Failed to delete file: {}", filename));

  // Save the updated card back to disk.
  Error error;
  const std::string& card_path = mc->GetPath();
  if (!card_path.empty() && !MemoryCardImage::SaveToFile(data, card_path.c_str(), &error))
    return ToolResult::Error(-2, fmt::format("Failed to save memory card: {}", error.GetDescription()));

  JsonWriter w;
  w.StartObject();
  w.KeyString("status", "deleted");
  w.KeyUint("slot", slot);
  w.KeyString("filename", filename);
  w.KeyUint("free_blocks", MemoryCardImage::GetFreeBlockCount(data));
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleExportMemoryCardSave(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (!args.contains("slot") || !args["slot"].is_number())
    return ToolResult::Error(-32602, "Missing 'slot' parameter");
  if (!args.contains("filename") || !args["filename"].is_string())
    return ToolResult::Error(-32602, "Missing 'filename' parameter");
  if (!args.contains("output_path") || !args["output_path"].is_string())
    return ToolResult::Error(-32602, "Missing 'output_path' parameter");

  const u32 slot = static_cast<u32>(args["slot"].get_uint());
  if (slot > 1)
    return ToolResult::Error(-32602, "Slot must be 0 or 1");

  MemoryCard* mc = Pad::GetMemoryCard(slot);
  if (!mc)
    return ToolResult::Error(-2, fmt::format("No memory card in slot {}", slot));

  MemoryCardImage::DataArray& data = mc->GetData();
  if (!MemoryCardImage::IsValid(data))
    return ToolResult::Error(-2, "Memory card data is not valid");

  const std::string filename = std::string(args["filename"].get_string());
  const std::string output_path = std::string(args["output_path"].get_string());

  const auto files = MemoryCardImage::EnumerateFiles(data, false);

  const MemoryCardImage::FileInfo* target = nullptr;
  for (const auto& fi : files)
  {
    if (fi.filename == filename)
    {
      target = &fi;
      break;
    }
  }

  if (!target)
    return ToolResult::Error(-2, fmt::format("File not found on memory card: {}", filename));

  Error error;
  if (!MemoryCardImage::ExportSave(&data, *target, output_path.c_str(), &error))
    return ToolResult::Error(-2, fmt::format("Failed to export save: {}", error.GetDescription()));

  JsonWriter w;
  w.StartObject();
  w.KeyString("status", "exported");
  w.KeyUint("slot", slot);
  w.KeyString("filename", filename);
  w.KeyString("output_path", output_path);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleImportMemoryCardSave(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (!args.contains("slot") || !args["slot"].is_number())
    return ToolResult::Error(-32602, "Missing 'slot' parameter");
  if (!args.contains("input_path") || !args["input_path"].is_string())
    return ToolResult::Error(-32602, "Missing 'input_path' parameter");

  const u32 slot = static_cast<u32>(args["slot"].get_uint());
  if (slot > 1)
    return ToolResult::Error(-32602, "Slot must be 0 or 1");

  MemoryCard* mc = Pad::GetMemoryCard(slot);
  if (!mc)
    return ToolResult::Error(-2, fmt::format("No memory card in slot {}", slot));

  MemoryCardImage::DataArray& data = mc->GetData();
  if (!MemoryCardImage::IsValid(data))
    return ToolResult::Error(-2, "Memory card data is not valid");

  const std::string input_path = std::string(args["input_path"].get_string());

  if (!FileSystem::FileExists(input_path.c_str()))
    return ToolResult::Error(-2, fmt::format("File not found: {}", input_path));

  Error error;
  if (!MemoryCardImage::ImportSave(&data, input_path.c_str(), &error))
    return ToolResult::Error(-2, fmt::format("Failed to import save: {}", error.GetDescription()));

  // Save the updated card back to disk.
  const std::string& card_path = mc->GetPath();
  if (!card_path.empty() && !MemoryCardImage::SaveToFile(data, card_path.c_str(), &error))
    return ToolResult::Error(-2, fmt::format("Failed to save memory card: {}", error.GetDescription()));

  JsonWriter w;
  w.StartObject();
  w.KeyString("status", "imported");
  w.KeyUint("slot", slot);
  w.KeyString("input_path", input_path);
  w.KeyUint("free_blocks", MemoryCardImage::GetFreeBlockCount(data));
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

// ============================================================================
// BATCH 6: Post-Processing (6 tools)
// ============================================================================

static ToolResult HandleListShaders([[maybe_unused]] const JsonValue& args)
{
  const auto shaders = PostProcessing::GetAvailableShaderNames();

  JsonWriter w;
  w.StartObject();
  w.KeyUint("count", static_cast<u32>(shaders.size()));

  w.Key("shaders");
  w.StartArray();
  for (const auto& [name, type] : shaders)
  {
    w.StartObject();
    w.KeyString("name", name);
    w.KeyString("type", PostProcessing::GetShaderTypeDisplayName(type));
    w.EndObject();
  }
  w.EndArray();

  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleGetShaderChain([[maybe_unused]] const JsonValue& args)
{
  auto lock = Core::GetSettingsLock();
  const SettingsInterface* si = Core::GetSettingsInterface();
  if (!si)
    return ToolResult::Error(-1, "Settings interface not available");

  const char* section = PostProcessing::Config::DISPLAY_CHAIN_SECTION;
  const bool enabled = PostProcessing::Config::IsEnabled(*si, section);
  const u32 stage_count = PostProcessing::Config::GetStageCount(*si, section);

  JsonWriter w;
  w.StartObject();
  w.KeyBool("enabled", enabled);
  w.KeyUint("stage_count", stage_count);

  w.Key("stages");
  w.StartArray();
  for (u32 i = 0; i < stage_count; i++)
  {
    const std::string shader_name = PostProcessing::Config::GetStageShaderName(*si, section, i);
    const bool stage_enabled = PostProcessing::Config::IsStageEnabled(*si, section, i);

    w.StartObject();
    w.KeyUint("index", i);
    w.KeyString("shader_name", shader_name);
    w.KeyBool("enabled", stage_enabled);
    w.EndObject();
  }
  w.EndArray();

  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleAddShader(const JsonValue& args)
{
  if (!args.contains("shader_name") || !args["shader_name"].is_string())
    return ToolResult::Error(-32602, "Missing 'shader_name' parameter");

  const std::string shader_name = std::string(args["shader_name"].get_string());

  auto lock = Core::GetSettingsLock();
  SettingsInterface* si = Core::GetBaseSettingsLayer();
  if (!si)
    return ToolResult::Error(-1, "Settings interface not available");

  const char* section = PostProcessing::Config::DISPLAY_CHAIN_SECTION;

  Error error;
  if (!PostProcessing::Config::AddStage(*si, section, shader_name, &error))
    return ToolResult::Error(-2, fmt::format("Failed to add shader: {}", error.GetDescription()));

  const u32 stage_count = PostProcessing::Config::GetStageCount(*si, section);

  lock.unlock();
  Host::CommitBaseSettingChanges();

  JsonWriter w;
  w.StartObject();
  w.KeyString("status", "added");
  w.KeyString("shader_name", shader_name);
  w.KeyUint("index", stage_count > 0 ? stage_count - 1 : 0);
  w.KeyUint("stage_count", stage_count);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleRemoveShader(const JsonValue& args)
{
  if (!args.contains("index") || !args["index"].is_number())
    return ToolResult::Error(-32602, "Missing 'index' parameter");

  const u32 index = static_cast<u32>(args["index"].get_uint());

  auto lock = Core::GetSettingsLock();
  SettingsInterface* si = Core::GetBaseSettingsLayer();
  if (!si)
    return ToolResult::Error(-1, "Settings interface not available");

  const char* section = PostProcessing::Config::DISPLAY_CHAIN_SECTION;
  const u32 stage_count = PostProcessing::Config::GetStageCount(*si, section);

  if (index >= stage_count)
    return ToolResult::Error(-32602, fmt::format("Invalid stage index {} (count: {})", index, stage_count));

  const std::string removed_name = PostProcessing::Config::GetStageShaderName(*si, section, index);
  PostProcessing::Config::RemoveStage(*si, section, index);
  const u32 new_stage_count = PostProcessing::Config::GetStageCount(*si, section);

  lock.unlock();
  Host::CommitBaseSettingChanges();

  JsonWriter w;
  w.StartObject();
  w.KeyString("status", "removed");
  w.KeyUint("index", index);
  w.KeyString("shader_name", removed_name);
  w.KeyUint("stage_count", new_stage_count);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleGetShaderOptions(const JsonValue& args)
{
  if (!args.contains("index") || !args["index"].is_number())
    return ToolResult::Error(-32602, "Missing 'index' parameter");

  const u32 index = static_cast<u32>(args["index"].get_uint());

  auto lock = Core::GetSettingsLock();
  const SettingsInterface* si = Core::GetSettingsInterface();
  if (!si)
    return ToolResult::Error(-1, "Settings interface not available");

  const char* section = PostProcessing::Config::DISPLAY_CHAIN_SECTION;
  const u32 stage_count = PostProcessing::Config::GetStageCount(*si, section);

  if (index >= stage_count)
    return ToolResult::Error(-32602, fmt::format("Invalid stage index {} (count: {})", index, stage_count));

  const std::string shader_name = PostProcessing::Config::GetStageShaderName(*si, section, index);
  const std::vector<PostProcessing::ShaderOption> options =
    PostProcessing::Config::GetStageOptions(*si, section, index);

  JsonWriter w;
  w.StartObject();
  w.KeyUint("index", index);
  w.KeyString("shader_name", shader_name);
  w.KeyUint("option_count", static_cast<u32>(options.size()));

  w.Key("options");
  w.StartArray();
  for (const auto& opt : options)
  {
    w.StartObject();
    w.KeyString("name", opt.name);
    w.KeyString("ui_name", opt.ui_name);

    const char* type_name = "invalid";
    switch (opt.type)
    {
      case PostProcessing::ShaderOption::Type::Bool:
        type_name = "bool";
        break;
      case PostProcessing::ShaderOption::Type::Int:
        type_name = "int";
        break;
      case PostProcessing::ShaderOption::Type::Float:
        type_name = "float";
        break;
      default:
        break;
    }
    w.KeyString("type", type_name);
    w.KeyUint("vector_size", opt.vector_size);

    // Write current value(s).
    w.Key("value");
    if (opt.vector_size <= 1)
    {
      if (opt.type == PostProcessing::ShaderOption::Type::Bool ||
          opt.type == PostProcessing::ShaderOption::Type::Int)
        w.Int(opt.value[0].int_value);
      else
        w.Double(static_cast<double>(opt.value[0].float_value));
    }
    else
    {
      w.StartArray();
      for (u32 i = 0; i < opt.vector_size; i++)
      {
        if (opt.type == PostProcessing::ShaderOption::Type::Bool ||
            opt.type == PostProcessing::ShaderOption::Type::Int)
          w.Int(opt.value[i].int_value);
        else
          w.Double(static_cast<double>(opt.value[i].float_value));
      }
      w.EndArray();
    }

    // Write default value(s).
    w.Key("default_value");
    if (opt.vector_size <= 1)
    {
      if (opt.type == PostProcessing::ShaderOption::Type::Bool ||
          opt.type == PostProcessing::ShaderOption::Type::Int)
        w.Int(opt.default_value[0].int_value);
      else
        w.Double(static_cast<double>(opt.default_value[0].float_value));
    }
    else
    {
      w.StartArray();
      for (u32 i = 0; i < opt.vector_size; i++)
      {
        if (opt.type == PostProcessing::ShaderOption::Type::Bool ||
            opt.type == PostProcessing::ShaderOption::Type::Int)
          w.Int(opt.default_value[i].int_value);
        else
          w.Double(static_cast<double>(opt.default_value[i].float_value));
      }
      w.EndArray();
    }

    // Write min/max for numeric types.
    if (opt.type == PostProcessing::ShaderOption::Type::Int ||
        opt.type == PostProcessing::ShaderOption::Type::Float)
    {
      w.Key("min_value");
      if (opt.vector_size <= 1)
      {
        if (opt.type == PostProcessing::ShaderOption::Type::Int)
          w.Int(opt.min_value[0].int_value);
        else
          w.Double(static_cast<double>(opt.min_value[0].float_value));
      }
      else
      {
        w.StartArray();
        for (u32 i = 0; i < opt.vector_size; i++)
        {
          if (opt.type == PostProcessing::ShaderOption::Type::Int)
            w.Int(opt.min_value[i].int_value);
          else
            w.Double(static_cast<double>(opt.min_value[i].float_value));
        }
        w.EndArray();
      }

      w.Key("max_value");
      if (opt.vector_size <= 1)
      {
        if (opt.type == PostProcessing::ShaderOption::Type::Int)
          w.Int(opt.max_value[0].int_value);
        else
          w.Double(static_cast<double>(opt.max_value[0].float_value));
      }
      else
      {
        w.StartArray();
        for (u32 i = 0; i < opt.vector_size; i++)
        {
          if (opt.type == PostProcessing::ShaderOption::Type::Int)
            w.Int(opt.max_value[i].int_value);
          else
            w.Double(static_cast<double>(opt.max_value[i].float_value));
        }
        w.EndArray();
      }

      w.Key("step_value");
      if (opt.vector_size <= 1)
      {
        if (opt.type == PostProcessing::ShaderOption::Type::Int)
          w.Int(opt.step_value[0].int_value);
        else
          w.Double(static_cast<double>(opt.step_value[0].float_value));
      }
      else
      {
        w.StartArray();
        for (u32 i = 0; i < opt.vector_size; i++)
        {
          if (opt.type == PostProcessing::ShaderOption::Type::Int)
            w.Int(opt.step_value[i].int_value);
          else
            w.Double(static_cast<double>(opt.step_value[i].float_value));
        }
        w.EndArray();
      }
    }

    if (!opt.category.empty())
      w.KeyString("category", opt.category);
    if (!opt.tooltip.empty())
      w.KeyString("tooltip", opt.tooltip);

    if (!opt.choice_options.empty())
    {
      w.Key("choices");
      w.StartArray();
      for (const auto& choice : opt.choice_options)
        w.String(choice);
      w.EndArray();
    }

    w.EndObject();
  }
  w.EndArray();

  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleSetShaderOption(const JsonValue& args)
{
  if (!args.contains("index") || !args["index"].is_number())
    return ToolResult::Error(-32602, "Missing 'index' parameter");
  if (!args.contains("option_name") || !args["option_name"].is_string())
    return ToolResult::Error(-32602, "Missing 'option_name' parameter");
  if (!args.contains("value"))
    return ToolResult::Error(-32602, "Missing 'value' parameter");

  const u32 index = static_cast<u32>(args["index"].get_uint());
  const std::string option_name = std::string(args["option_name"].get_string());

  auto lock = Core::GetSettingsLock();
  SettingsInterface* si = Core::GetBaseSettingsLayer();
  if (!si)
    return ToolResult::Error(-1, "Settings interface not available");

  const char* section = PostProcessing::Config::DISPLAY_CHAIN_SECTION;
  const u32 stage_count = PostProcessing::Config::GetStageCount(*si, section);

  if (index >= stage_count)
    return ToolResult::Error(-32602, fmt::format("Invalid stage index {} (count: {})", index, stage_count));

  std::vector<PostProcessing::ShaderOption> options =
    PostProcessing::Config::GetStageOptions(*si, section, index);

  PostProcessing::ShaderOption* target_option = nullptr;
  for (auto& opt : options)
  {
    if (opt.name == option_name)
    {
      target_option = &opt;
      break;
    }
  }

  if (!target_option)
    return ToolResult::Error(-2, fmt::format("Option not found: {}", option_name));

  // Update the option value based on its type.
  const JsonValue& value_arg = args["value"];
  if (value_arg.is_array())
  {
    const u32 count = std::min(static_cast<u32>(value_arg.size()), target_option->vector_size);
    for (u32 i = 0; i < count; i++)
    {
      if (target_option->type == PostProcessing::ShaderOption::Type::Float)
        target_option->value[i].float_value = static_cast<float>(value_arg[static_cast<size_t>(i)].get_float());
      else
        target_option->value[i].int_value = static_cast<s32>(value_arg[static_cast<size_t>(i)].get_int());
    }
  }
  else
  {
    if (target_option->type == PostProcessing::ShaderOption::Type::Bool)
    {
      if (value_arg.is_bool())
        target_option->value[0].int_value = value_arg.get_bool() ? 1 : 0;
      else
        target_option->value[0].int_value = static_cast<s32>(value_arg.get_int());
    }
    else if (target_option->type == PostProcessing::ShaderOption::Type::Float)
    {
      target_option->value[0].float_value = static_cast<float>(value_arg.get_float());
    }
    else if (target_option->type == PostProcessing::ShaderOption::Type::Int)
    {
      target_option->value[0].int_value = static_cast<s32>(value_arg.get_int());
    }
  }

  PostProcessing::Config::SetStageOption(*si, section, index, *target_option);

  lock.unlock();
  Host::CommitBaseSettingChanges();

  JsonWriter w;
  w.StartObject();
  w.KeyString("status", "updated");
  w.KeyUint("index", index);
  w.KeyString("option_name", option_name);

  // Echo back the new value.
  w.Key("value");
  if (target_option->vector_size <= 1)
  {
    if (target_option->type == PostProcessing::ShaderOption::Type::Float)
      w.Double(static_cast<double>(target_option->value[0].float_value));
    else
      w.Int(target_option->value[0].int_value);
  }
  else
  {
    w.StartArray();
    for (u32 i = 0; i < target_option->vector_size; i++)
    {
      if (target_option->type == PostProcessing::ShaderOption::Type::Float)
        w.Double(static_cast<double>(target_option->value[i].float_value));
      else
        w.Int(target_option->value[i].int_value);
    }
    w.EndArray();
  }

  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

// ============================================================================
// BATCH 7: Media Capture (3 tools)
// ============================================================================

static ToolResult HandleStartCapture(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  std::string path;
  if (args.contains("path") && args["path"].is_string())
    path = std::string(args["path"].get_string());

  if (!System::StartMediaCapture(std::move(path)))
    return ToolResult::Error(-3, "Failed to start media capture");

  JsonWriter w;
  w.StartObject();
  w.KeyString("status", "started");

  MediaCapture* cap = System::GetMediaCapture();
  if (cap)
  {
    w.KeyString("path", cap->GetPath());
    w.KeyBool("capturing_audio", cap->IsCapturingAudio());
    w.KeyBool("capturing_video", cap->IsCapturingVideo());
  }

  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleStopCapture([[maybe_unused]] const JsonValue& args)
{
  System::StopMediaCapture();

  JsonWriter w;
  w.StartObject();
  w.KeyString("status", "stopped");
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleGetCaptureStatus([[maybe_unused]] const JsonValue& args)
{
  MediaCapture* cap = System::GetMediaCapture();

  JsonWriter w;
  w.StartObject();

  if (!cap)
  {
    w.KeyBool("active", false);
  }
  else
  {
    w.KeyBool("active", true);
    w.KeyString("path", cap->GetPath());
    w.KeyBool("capturing_audio", cap->IsCapturingAudio());
    w.KeyBool("capturing_video", cap->IsCapturingVideo());
    w.KeyUint("video_width", cap->GetVideoWidth());
    w.KeyUint("video_height", cap->GetVideoHeight());
    w.KeyDouble("video_fps", cap->GetVideoFPS());
    w.KeyInt("elapsed_time_seconds", static_cast<s64>(cap->GetElapsedTime()));
    w.KeyDouble("capture_thread_usage", cap->GetCaptureThreadUsage());
  }

  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

// ============================================================================
// BATCH 8: Memory Scanner (7 tools)
// ============================================================================

// ---- Helpers ----

static MemoryAccessSize ParseMemoryAccessSize(std::string_view s)
{
  if (s == "byte")
    return MemoryAccessSize::Byte;
  if (s == "halfword")
    return MemoryAccessSize::HalfWord;
  return MemoryAccessSize::Word;
}

static const char* MemoryAccessSizeToString(MemoryAccessSize size)
{
  switch (size)
  {
    case MemoryAccessSize::Byte:
      return "byte";
    case MemoryAccessSize::HalfWord:
      return "halfword";
    case MemoryAccessSize::Word:
      return "word";
    default:
      return "word";
  }
}

static MemoryScan::Operator ParseScanOperator(std::string_view s)
{
  if (s == "equal")
    return MemoryScan::Operator::Equal;
  if (s == "not_equal")
    return MemoryScan::Operator::NotEqual;
  if (s == "less_than")
    return MemoryScan::Operator::LessThan;
  if (s == "less_equal")
    return MemoryScan::Operator::LessEqual;
  if (s == "greater_than")
    return MemoryScan::Operator::GreaterThan;
  if (s == "greater_equal")
    return MemoryScan::Operator::GreaterEqual;
  if (s == "any")
    return MemoryScan::Operator::Any;
  if (s == "changed")
    return MemoryScan::Operator::ChangedBy;
  if (s == "decreased")
    return MemoryScan::Operator::DecreasedBy;
  if (s == "increased")
    return MemoryScan::Operator::IncreasedBy;
  if (s == "less_than_last")
    return MemoryScan::Operator::LessThanLast;
  if (s == "less_equal_last")
    return MemoryScan::Operator::LessEqualLast;
  if (s == "greater_than_last")
    return MemoryScan::Operator::GreaterThanLast;
  if (s == "greater_equal_last")
    return MemoryScan::Operator::GreaterEqualLast;
  if (s == "not_equal_last")
    return MemoryScan::Operator::NotEqualLast;
  if (s == "equal_last")
    return MemoryScan::Operator::EqualLast;
  return MemoryScan::Operator::Equal;
}

// ---- Tool 1: memory_scan_start ----

static ToolResult HandleMemoryScanStart(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  // Parse value - accept integer or hex string
  u32 value = 0;
  if (args.contains("value"))
  {
    if (args["value"].is_number_integer())
      value = static_cast<u32>(args["value"].get_uint());
    else if (args["value"].is_string())
    {
      const std::string_view val_str = args["value"].get_string();
      if (val_str.size() > 2 && (val_str[0] == '0' && (val_str[1] == 'x' || val_str[1] == 'X')))
        value = StringUtil::FromChars<u32>(val_str.substr(2), 16).value_or(0);
      else
        value = StringUtil::FromChars<u32>(val_str).value_or(0);
    }
  }

  // Parse size
  MemoryAccessSize size = MemoryAccessSize::Word;
  if (args.contains("size") && args["size"].is_string())
    size = ParseMemoryAccessSize(args["size"].get_string());

  // Parse operator
  MemoryScan::Operator op = MemoryScan::Operator::Equal;
  if (args.contains("operator") && args["operator"].is_string())
    op = ParseScanOperator(args["operator"].get_string());

  // Parse signed
  bool is_signed = false;
  if (args.contains("signed") && args["signed"].is_bool())
    is_signed = args["signed"].get_bool();

  // Configure scan
  s_memory_scan.SetValue(value);
  s_memory_scan.SetSize(size);
  s_memory_scan.SetOperator(op);
  s_memory_scan.SetValueSigned(is_signed);

  // Parse optional start/end addresses
  if (args.contains("start") && args["start"].is_string())
  {
    const std::string_view addr_str = args["start"].get_string();
    u32 address = 0;
    if (addr_str.size() > 2 && (addr_str[0] == '0' && (addr_str[1] == 'x' || addr_str[1] == 'X')))
      address = StringUtil::FromChars<u32>(addr_str.substr(2), 16).value_or(0);
    else
      address = StringUtil::FromChars<u32>(addr_str).value_or(0);
    s_memory_scan.SetStartAddress(address);
  }

  if (args.contains("end") && args["end"].is_string())
  {
    const std::string_view addr_str = args["end"].get_string();
    u32 address = 0x200000;
    if (addr_str.size() > 2 && (addr_str[0] == '0' && (addr_str[1] == 'x' || addr_str[1] == 'X')))
      address = StringUtil::FromChars<u32>(addr_str.substr(2), 16).value_or(0x200000);
    else
      address = StringUtil::FromChars<u32>(addr_str).value_or(0x200000);
    s_memory_scan.SetEndAddress(address);
  }

  // Perform initial scan
  s_memory_scan.ResetSearch();
  s_memory_scan.Search();

  JsonWriter w;
  w.StartObject();
  w.KeyString("status", "completed");
  w.KeyUint("result_count", s_memory_scan.GetResultCount());
  w.KeyString("size", MemoryAccessSizeToString(size));
  w.KeyUint("value", value);
  w.KeyBool("signed", is_signed);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

// ---- Tool 2: memory_scan_refine ----

static ToolResult HandleMemoryScanRefine(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (s_memory_scan.GetResultCount() == 0)
    return ToolResult::Error(-2, "No active scan results to refine. Run memory_scan_start first.");

  // Update value if provided
  if (args.contains("value"))
  {
    if (args["value"].is_number_integer())
      s_memory_scan.SetValue(static_cast<u32>(args["value"].get_uint()));
    else if (args["value"].is_string())
    {
      const std::string_view val_str = args["value"].get_string();
      u32 value = 0;
      if (val_str.size() > 2 && (val_str[0] == '0' && (val_str[1] == 'x' || val_str[1] == 'X')))
        value = StringUtil::FromChars<u32>(val_str.substr(2), 16).value_or(0);
      else
        value = StringUtil::FromChars<u32>(val_str).value_or(0);
      s_memory_scan.SetValue(value);
    }
  }

  // Parse operator
  if (args.contains("operator") && args["operator"].is_string())
    s_memory_scan.SetOperator(ParseScanOperator(args["operator"].get_string()));

  s_memory_scan.SearchAgain();

  JsonWriter w;
  w.StartObject();
  w.KeyString("status", "refined");
  w.KeyUint("result_count", s_memory_scan.GetResultCount());
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

// ---- Tool 3: memory_scan_results ----

static ToolResult HandleMemoryScanResults(const JsonValue& args)
{
  u32 offset = 0;
  u32 count = 100;

  if (args.contains("offset") && args["offset"].is_number_integer())
    offset = static_cast<u32>(args["offset"].get_uint());
  if (args.contains("count") && args["count"].is_number_integer())
    count = static_cast<u32>(args["count"].get_uint());

  const u32 total = s_memory_scan.GetResultCount();
  const u32 start = std::min(offset, total);
  const u32 end = std::min(start + count, total);

  // Update values before returning
  if (total > 0)
    s_memory_scan.UpdateResultsValues();

  const MemoryScan::ResultVector& results = s_memory_scan.GetResults();

  JsonWriter w;
  w.StartObject();
  w.KeyUint("total_results", total);
  w.KeyUint("offset", start);
  w.KeyUint("count", end - start);

  w.Key("results");
  w.StartArray();
  for (u32 i = start; i < end; i++)
  {
    const MemoryScan::Result& r = results[i];
    w.StartObject();
    w.KeyString("address", FormatHex32(r.address));
    w.KeyUint("value", r.value);
    w.KeyUint("last_value", r.last_value);
    w.KeyUint("first_value", r.first_value);
    w.KeyBool("value_changed", r.value_changed);
    w.EndObject();
  }
  w.EndArray();

  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

// ---- Tool 4: memory_scan_reset ----

static ToolResult HandleMemoryScanReset([[maybe_unused]] const JsonValue& args)
{
  s_memory_scan.ResetSearch();

  JsonWriter w;
  w.StartObject();
  w.KeyString("status", "reset");
  w.KeyUint("result_count", 0);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

// ---- Tool 5: add_memory_watch ----

static ToolResult HandleAddMemoryWatch(const JsonValue& args)
{
  if (!args.contains("address"))
    return ToolResult::Error(-32602, "Missing 'address' parameter");

  // Parse address
  u32 address = 0;
  if (args["address"].is_string())
  {
    const std::string_view addr_str = args["address"].get_string();
    if (addr_str.size() > 2 && (addr_str[0] == '0' && (addr_str[1] == 'x' || addr_str[1] == 'X')))
      address = StringUtil::FromChars<u32>(addr_str.substr(2), 16).value_or(0);
    else
      address = StringUtil::FromChars<u32>(addr_str).value_or(0);
  }
  else if (args["address"].is_number_integer())
  {
    address = static_cast<u32>(args["address"].get_uint());
  }
  else
  {
    return ToolResult::Error(-32602, "Invalid 'address' parameter");
  }

  // Parse size
  MemoryAccessSize size = MemoryAccessSize::Word;
  if (args.contains("size") && args["size"].is_string())
    size = ParseMemoryAccessSize(args["size"].get_string());

  // Parse description
  std::string description;
  if (args.contains("description") && args["description"].is_string())
    description = std::string(args["description"].get_string());
  else
    description = fmt::format("Watch @ {}", FormatHex32(address));

  // Parse signed
  bool is_signed = false;
  if (args.contains("signed") && args["signed"].is_bool())
    is_signed = args["signed"].get_bool();

  // Parse freeze
  bool freeze = false;
  if (args.contains("freeze") && args["freeze"].is_bool())
    freeze = args["freeze"].get_bool();

  if (!s_memory_watch_list.AddEntry(std::move(description), address, size, is_signed, freeze))
    return ToolResult::Error(-3, fmt::format("Failed to add watch at address {}", FormatHex32(address)));

  JsonWriter w;
  w.StartObject();
  w.KeyString("status", "added");
  w.KeyString("address", FormatHex32(address));
  w.KeyString("size", MemoryAccessSizeToString(size));
  w.KeyBool("freeze", freeze);
  w.KeyUint("total_watches", s_memory_watch_list.GetEntryCount());
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

// ---- Tool 6: remove_memory_watch ----

static ToolResult HandleRemoveMemoryWatch(const JsonValue& args)
{
  if (!args.contains("address"))
    return ToolResult::Error(-32602, "Missing 'address' parameter");

  // Parse address
  u32 address = 0;
  if (args["address"].is_string())
  {
    const std::string_view addr_str = args["address"].get_string();
    if (addr_str.size() > 2 && (addr_str[0] == '0' && (addr_str[1] == 'x' || addr_str[1] == 'X')))
      address = StringUtil::FromChars<u32>(addr_str.substr(2), 16).value_or(0);
    else
      address = StringUtil::FromChars<u32>(addr_str).value_or(0);
  }
  else if (args["address"].is_number_integer())
  {
    address = static_cast<u32>(args["address"].get_uint());
  }
  else
  {
    return ToolResult::Error(-32602, "Invalid 'address' parameter");
  }

  if (!s_memory_watch_list.RemoveEntryByAddress(address))
    return ToolResult::Error(-3, fmt::format("No watch found at address {}", FormatHex32(address)));

  JsonWriter w;
  w.StartObject();
  w.KeyString("status", "removed");
  w.KeyString("address", FormatHex32(address));
  w.KeyUint("total_watches", s_memory_watch_list.GetEntryCount());
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

// ---- Tool 7: list_memory_watches ----

static ToolResult HandleListMemoryWatches([[maybe_unused]] const JsonValue& args)
{
  // Update values before listing
  s_memory_watch_list.UpdateValues();

  const MemoryWatchList::EntryVector& entries = s_memory_watch_list.GetEntries();

  JsonWriter w;
  w.StartObject();
  w.KeyUint("count", s_memory_watch_list.GetEntryCount());

  w.Key("watches");
  w.StartArray();
  for (const MemoryWatchList::Entry& entry : entries)
  {
    w.StartObject();
    w.KeyString("description", entry.description);
    w.KeyString("address", FormatHex32(entry.address));
    w.KeyUint("value", entry.value);
    w.KeyString("size", MemoryAccessSizeToString(entry.size));
    w.KeyBool("is_signed", entry.is_signed);
    w.KeyBool("freeze", entry.freeze);
    w.KeyBool("changed", entry.changed);
    w.EndObject();
  }
  w.EndArray();

  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

// ============================================================================
// BATCH 9: Hotkeys & System (4 tools)
// ============================================================================

// ---- Tool 1: list_hotkeys ----

static ToolResult HandleListHotkeys(const JsonValue& args)
{
  const std::span<const HotkeyInfo> hotkeys = Core::GetHotkeyList();

  std::string_view category_filter;
  if (args.contains("category") && args["category"].is_string())
    category_filter = args["category"].get_string();

  JsonWriter w;
  w.StartObject();

  w.Key("hotkeys");
  w.StartArray();

  u32 count = 0;
  for (const HotkeyInfo& hk : hotkeys)
  {
    if (!category_filter.empty() && !StringUtil::EqualNoCase(hk.category, category_filter))
      continue;

    w.StartObject();
    w.KeyString("name", hk.name);
    w.KeyString("category", hk.category);
    w.KeyString("display_name", hk.display_name);
    w.EndObject();
    count++;
  }
  w.EndArray();

  w.KeyUint("count", count);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

// ---- Tool 2: trigger_hotkey ----

static ToolResult HandleTriggerHotkey(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (!args.contains("name") || !args["name"].is_string())
    return ToolResult::Error(-32602, "Missing 'name' parameter");

  const std::string_view name = args["name"].get_string();
  const std::span<const HotkeyInfo> hotkeys = Core::GetHotkeyList();

  for (const HotkeyInfo& hk : hotkeys)
  {
    if (StringUtil::EqualNoCase(hk.name, name))
    {
      if (!hk.handler)
        return ToolResult::Error(-3, fmt::format("Hotkey '{}' has no handler", name));

      // Simulate press then release
      hk.handler(1);
      hk.handler(0);

      JsonWriter w;
      w.StartObject();
      w.KeyString("status", "triggered");
      w.KeyString("name", hk.name);
      w.KeyString("display_name", hk.display_name);
      w.EndObject();
      return ToolResult{w.TakeOutput()};
    }
  }

  return ToolResult::Error(-32602, fmt::format("Unknown hotkey '{}'", name));
}


// ============================================================================
// BATCH 10: Achievements & Disc Info (2 tools)
// ============================================================================

// ---- Tool 1: get_achievements_details ----

static ToolResult HandleGetAchievementsDetails([[maybe_unused]] const JsonValue& args)
{
  JsonWriter w;
  w.StartObject();

  w.KeyBool("active", Achievements::IsActive());

  if (!Achievements::IsActive())
  {
    w.EndObject();
    return ToolResult{w.TakeOutput()};
  }

  w.KeyBool("logged_in", Achievements::IsLoggedIn());
  w.KeyBool("hardcore_mode", Achievements::IsHardcoreModeActive());

  if (Achievements::IsLoggedIn())
  {
    w.KeyString("username", Achievements::GetLoggedInUserName());

    auto lock = Achievements::GetLock();
    w.KeyString("user_badge_path", Achievements::GetLoggedInUserBadgePath());
    w.KeyString("points_summary", std::string(Achievements::GetLoggedInUserPointsSummary()));
  }

  w.KeyBool("has_active_game", Achievements::HasActiveGame());
  w.KeyUint("game_id", Achievements::GetGameID());

  if (Achievements::HasActiveGame())
  {
    auto lock = Achievements::GetLock();

    w.KeyString("game_title", Achievements::GetGameTitle());
    w.KeyString("game_path", Achievements::GetGamePath());
    w.KeyString("game_icon_url", Achievements::GetGameIconURL());
    w.KeyBool("has_achievements", Achievements::HasAchievements());
    w.KeyBool("has_leaderboards", Achievements::HasLeaderboards());
    w.KeyBool("has_rich_presence", Achievements::HasRichPresence());

    if (Achievements::HasRichPresence())
      w.KeyString("rich_presence", Achievements::GetRichPresenceString());
  }

  w.KeyUint("pending_unlock_count", Achievements::GetPendingUnlockCount());

  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

// ---- Tool 2: get_disc_info ----

static const char* TrackModeToString(CDImage::TrackMode mode)
{
  switch (mode)
  {
    case CDImage::TrackMode::Audio:
      return "Audio";
    case CDImage::TrackMode::Mode1:
      return "Mode1";
    case CDImage::TrackMode::Mode1Raw:
      return "Mode1Raw";
    case CDImage::TrackMode::Mode2:
      return "Mode2";
    case CDImage::TrackMode::Mode2Form1:
      return "Mode2Form1";
    case CDImage::TrackMode::Mode2Form2:
      return "Mode2Form2";
    case CDImage::TrackMode::Mode2FormMix:
      return "Mode2FormMix";
    case CDImage::TrackMode::Mode2Raw:
      return "Mode2Raw";
    default:
      return "Unknown";
  }
}

static ToolResult HandleGetDiscInfo([[maybe_unused]] const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (!CDROM::HasMedia())
    return ToolResult::Error(-2, "No disc inserted");

  const CDImage* media = CDROM::GetMedia();
  if (!media)
    return ToolResult::Error(-2, "No disc media available");

  JsonWriter w;
  w.StartObject();

  w.KeyString("media_path", CDROM::GetMediaPath());
  w.KeyString("disc_region", Settings::GetDiscRegionName(CDROM::GetDiscRegion()));
  w.KeyBool("is_ps1_disc", CDROM::IsMediaPS1Disc());
  w.KeyBool("is_audio_cd", CDROM::IsMediaAudioCD());
  w.KeyUint("track_count", media->GetTrackCount());

  if (System::HasMediaSubImages())
  {
    w.KeyUint("sub_image_count", System::GetMediaSubImageCount());
    w.KeyUint("current_sub_image", System::GetMediaSubImageIndex());
  }

  w.Key("tracks");
  w.StartArray();

  const u32 track_count = media->GetTrackCount();
  for (u32 i = 1; i <= track_count; i++)
  {
    const CDImage::Track& track = media->GetTrack(i);

    w.StartObject();
    w.KeyUint("track_number", track.track_number);
    w.KeyString("mode", TrackModeToString(track.mode));
    w.KeyUint("start_lba", track.start_lba);
    w.KeyUint("length", track.length);
    w.KeyUint("bytes_per_sector", CDImage::GetBytesPerSector(track.mode));
    w.KeyBool("is_data", track.control.data);
    w.EndObject();
  }
  w.EndArray();

  w.EndObject();
  return ToolResult{w.TakeOutput()};
}


// ---- ROM Hacking tools ----

static ToolResult HandleListDiscFiles(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  const CDImage* media = CDROM::GetMedia();
  if (!media)
    return ToolResult::Error(-1, "No disc loaded");

  const std::string_view path =
    (args.contains("path") && args["path"].is_string()) ? args["path"].get_string() : "/";
  const bool recursive =
    (args.contains("recursive") && args["recursive"].is_bool()) ? args["recursive"].get_bool() : false;

  IsoReader iso;
  Error error;
  if (!iso.Open(const_cast<CDImage*>(media), 1, &error))
    return ToolResult::Error(-2, fmt::format("Failed to open ISO filesystem: {}", error.GetDescription()));

  JsonWriter w;
  w.StartObject();
  w.KeyString("path", path);

  u32 count = 0;

  w.Key("entries");
  w.StartArray();

  std::function<void(std::string_view)> list_dir = [&](std::string_view dir_path) {
    auto entries = iso.GetEntriesInDirectory(dir_path, &error);
    for (const auto& [name, de] : entries)
    {
      w.StartObject();

      if (recursive && dir_path != "/" && !dir_path.empty())
        w.KeyString("name", fmt::format("{}/{}", dir_path, name));
      else
        w.KeyString("name", name);

      w.KeyBool("is_directory", de.IsDirectory());
      w.KeyUint("size", de.length_le);
      w.KeyUint("lba", de.location_le);
      w.KeyUint("sectors", de.GetSizeInSectors());
      w.EndObject();
      count++;

      if (recursive && de.IsDirectory())
      {
        std::string subpath =
          (dir_path == "/" || dir_path.empty()) ? fmt::format("/{}", name) : fmt::format("{}/{}", dir_path, name);
        list_dir(subpath);
      }
    }
  };

  list_dir(path);

  w.EndArray();
  w.KeyUint("count", count);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleReadDiscFile(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  const CDImage* media = CDROM::GetMedia();
  if (!media)
    return ToolResult::Error(-1, "No disc loaded");

  if (!args.contains("path") || !args["path"].is_string())
    return ToolResult::Error(-32602, "Missing required 'path' parameter");

  const std::string disc_path = std::string(args["path"].get_string());

  IsoReader iso;
  Error error;
  if (!iso.Open(const_cast<CDImage*>(media), 1, &error))
    return ToolResult::Error(-2, fmt::format("Failed to open ISO filesystem: {}", error.GetDescription()));

  std::vector<u8> data;
  if (!iso.ReadFile(disc_path, &data, IsoReader::ReadMode::Data, &error))
    return ToolResult::Error(-2, fmt::format("Failed to read '{}': {}", disc_path, error.GetDescription()));

  static constexpr size_t MAX_FILE_SIZE = 16u * 1024u * 1024u;
  if (data.size() > MAX_FILE_SIZE)
    return ToolResult::Error(-3,
                             fmt::format("File too large ({} bytes, limit is {} bytes)", data.size(), MAX_FILE_SIZE));

  const std::string output_path = GetMCPTempFilePath("disc_file", "bin");
  if (!FileSystem::WriteBinaryFile(output_path.c_str(), data.data(), data.size(), &error))
    return ToolResult::Error(-2, fmt::format("Failed to write temp file: {}", error.GetDescription()));

  JsonWriter w;
  w.StartObject();
  w.KeyString("disc_path", disc_path);
  w.KeyUint("size", static_cast<u32>(data.size()));
  w.KeyString("output_path", output_path);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleReadDiscSectors(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  const CDImage* media = CDROM::GetMedia();
  if (!media)
    return ToolResult::Error(-1, "No disc loaded");

  if (!args.contains("lba") || !args["lba"].is_number_integer())
    return ToolResult::Error(-32602, "Missing required 'lba' parameter");

  const u32 lba = static_cast<u32>(args["lba"].get_int());

  u32 count = 1;
  if (args.contains("count") && args["count"].is_number_integer())
    count = static_cast<u32>(args["count"].get_int());
  if (count < 1)
    count = 1;
  if (count > 128)
    count = 128;

  CDImage* mutable_media = const_cast<CDImage*>(media);

  if (!mutable_media->Seek(lba))
    return ToolResult::Error(-2, fmt::format("Failed to seek to LBA {}", lba));

  const u32 sector_size = CDImage::RAW_SECTOR_SIZE;
  const u32 total_bytes = sector_size * count;
  std::vector<u8> buffer(total_bytes);

  for (u32 i = 0; i < count; i++)
  {
    if (!mutable_media->ReadRawSector(buffer.data() + i * sector_size, nullptr))
      return ToolResult::Error(-2, fmt::format("Failed to read sector at LBA {}", lba + i));
  }

  const std::string output_path = GetMCPTempFilePath("disc_sectors", "bin");
  Error error;
  if (!FileSystem::WriteBinaryFile(output_path.c_str(), buffer.data(), buffer.size(), &error))
    return ToolResult::Error(-2, fmt::format("Failed to write temp file: {}", error.GetDescription()));

  JsonWriter w;
  w.StartObject();
  w.KeyUint("lba", lba);
  w.KeyUint("count", count);
  w.KeyUint("sector_size", sector_size);
  w.KeyUint("total_bytes", total_bytes);
  w.KeyString("output_path", output_path);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleReadVramRegion(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (!args.contains("x") || !args.contains("y") || !args.contains("width") || !args.contains("height"))
    return ToolResult::Error(-32602, "Missing required parameters (x, y, width, height)");

  const u32 x = static_cast<u32>(args["x"].get_int());
  const u32 y = static_cast<u32>(args["y"].get_int());
  const u32 width = static_cast<u32>(args["width"].get_int());
  const u32 height = static_cast<u32>(args["height"].get_int());

  static constexpr u32 VW = VRAM_WIDTH;
  static constexpr u32 VH = VRAM_HEIGHT;

  if (width == 0 || height == 0)
    return ToolResult::Error(-32602, "width and height must be greater than 0");
  if ((x + width) > VW || (y + height) > VH)
    return ToolResult::Error(-32602,
                             fmt::format("Region ({},{})+({}x{}) exceeds VRAM bounds ({}x{})", x, y, width, height,
                                         VW, VH));

  std::string format = "png";
  if (args.contains("format") && args["format"].is_string())
    format = std::string(args["format"].get_string());

  Error error;

  if (format == "raw")
  {
    const u32 raw_size = width * height * sizeof(u16);
    std::vector<u8> raw_data(raw_size);
    for (u32 row = 0; row < height; row++)
      std::memcpy(raw_data.data() + row * width * sizeof(u16), &g_vram[(y + row) * VW + x],
                  width * sizeof(u16));

    const std::string output_path = GetMCPTempFilePath("vram_region", "bin");
    if (!FileSystem::WriteBinaryFile(output_path.c_str(), raw_data.data(), raw_data.size(), &error))
      return ToolResult::Error(-2, fmt::format("Failed to write temp file: {}", error.GetDescription()));

    JsonWriter w;
    w.StartObject();
    w.KeyUint("x", x);
    w.KeyUint("y", y);
    w.KeyUint("width", width);
    w.KeyUint("height", height);
    w.KeyString("format", "raw");
    w.KeyString("output_path", output_path);
    w.EndObject();
    return ToolResult{w.TakeOutput()};
  }
  else if (format == "png")
  {
    Image img(width, height, ImageFormat::RGBA8);
    u8* dst_pixels = img.GetPixels();
    const u32 dst_pitch = img.GetPitch();

    for (u32 row = 0; row < height; row++)
    {
      u8* dst_row = dst_pixels + row * dst_pitch;
      for (u32 col = 0; col < width; col++)
      {
        const u16 pixel = g_vram[(y + row) * VW + (x + col)];
        const u8 r5 = static_cast<u8>(pixel & 0x1F);
        const u8 g5 = static_cast<u8>((pixel >> 5) & 0x1F);
        const u8 b5 = static_cast<u8>((pixel >> 10) & 0x1F);
        dst_row[col * 4 + 0] = (r5 << 3) | (r5 >> 2);
        dst_row[col * 4 + 1] = (g5 << 3) | (g5 >> 2);
        dst_row[col * 4 + 2] = (b5 << 3) | (b5 >> 2);
        dst_row[col * 4 + 3] = 0xFF;
      }
    }

    const std::string output_path = GetMCPTempFilePath("vram_region", "png");
    if (!img.SaveToFile(output_path.c_str(), Image::DEFAULT_SAVE_QUALITY, &error))
      return ToolResult::Error(-2, fmt::format("Failed to save PNG: {}", error.GetDescription()));

    JsonWriter w;
    w.StartObject();
    w.KeyUint("x", x);
    w.KeyUint("y", y);
    w.KeyUint("width", width);
    w.KeyUint("height", height);
    w.KeyString("format", "png");
    w.KeyString("output_path", output_path);
    w.EndObject();
    return ToolResult{w.TakeOutput()};
  }
  else
  {
    return ToolResult::Error(-32602, "Invalid format. Use 'png' or 'raw'.");
  }
}

static ToolResult HandleWriteVramRegion(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (!args.contains("x") || !args.contains("y") || !args.contains("width") || !args.contains("height"))
    return ToolResult::Error(-32602, "Missing required parameters (x, y, width, height)");

  if (!args.contains("input_path") || !args["input_path"].is_string())
    return ToolResult::Error(-32602, "Missing required 'input_path' parameter");

  const u32 x = static_cast<u32>(args["x"].get_int());
  const u32 y = static_cast<u32>(args["y"].get_int());
  const u32 width = static_cast<u32>(args["width"].get_int());
  const u32 height = static_cast<u32>(args["height"].get_int());

  static constexpr u32 VW = VRAM_WIDTH;
  static constexpr u32 VH = VRAM_HEIGHT;

  if (width == 0 || height == 0)
    return ToolResult::Error(-32602, "width and height must be greater than 0");
  if ((x + width) > VW || (y + height) > VH)
    return ToolResult::Error(-32602,
                             fmt::format("Region ({},{})+({}x{}) exceeds VRAM bounds ({}x{})", x, y, width, height,
                                         VW, VH));

  const std::string input_path = std::string(args["input_path"].get_string());

  std::string format = "raw";
  if (args.contains("format") && args["format"].is_string())
    format = std::string(args["format"].get_string());

  if (format != "raw")
    return ToolResult::Error(-32602, "Currently only 'raw' format is supported for writing");

  Error error;
  auto file_data = FileSystem::ReadBinaryFile(input_path.c_str(), &error);
  if (!file_data.has_value())
    return ToolResult::Error(-2,
                             fmt::format("Failed to read input file '{}': {}", input_path, error.GetDescription()));

  const u32 expected_size = width * height * sizeof(u16);
  if (file_data->size() != expected_size)
  {
    return ToolResult::Error(-3, fmt::format("File size mismatch: expected {} bytes ({}x{}x2), got {}", expected_size,
                                             width, height, file_data->size()));
  }

  // Write raw u16 data row-by-row into g_vram.
  const u16* src = reinterpret_cast<const u16*>(file_data->data());
  for (u32 row = 0; row < height; row++)
    std::memcpy(&g_vram[(y + row) * VW + x], src + row * width, width * sizeof(u16));

  JsonWriter w;
  w.StartObject();
  w.KeyUint("x", x);
  w.KeyUint("y", y);
  w.KeyUint("width", width);
  w.KeyUint("height", height);
  w.KeyUint("bytes_written", expected_size);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleSnapshotMemory(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  u32 virt_addr = 0x80000000u;
  if (args.contains("address"))
  {
    const auto parsed = ParseAddress(args["address"]);
    if (!parsed.has_value())
      return ToolResult::Error(-32602, "Invalid 'address' parameter");
    virt_addr = parsed.value();
  }

  const u32 phys_addr = virt_addr & 0x1FFFFFFFu;

  if (phys_addr >= Bus::g_ram_size)
    return ToolResult::Error(-3, fmt::format("Address {} is out of RAM range (RAM size: {} bytes)",
                                             FormatHex32(virt_addr), Bus::g_ram_size));

  u32 snap_size = Bus::g_ram_size - phys_addr;
  if (args.contains("size") && args["size"].is_number_integer())
  {
    const u32 requested = static_cast<u32>(args["size"].get_int());
    if (requested == 0)
      return ToolResult::Error(-32602, "'size' must be greater than 0");
    if (phys_addr + requested > Bus::g_ram_size)
      return ToolResult::Error(-3, fmt::format("Requested size {} would exceed RAM bounds", requested));
    snap_size = requested;
  }

  s_memory_snapshot.resize(snap_size);
  std::memcpy(s_memory_snapshot.data(), Bus::g_ram + phys_addr, snap_size);
  s_snapshot_base_address = phys_addr;
  s_snapshot_size = snap_size;

  JsonWriter w;
  w.StartObject();
  w.KeyString("address", FormatHex32(virt_addr));
  w.KeyString("physical_address", FormatHex32(phys_addr));
  w.KeyUint("size", snap_size);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleDiffMemory(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  if (s_memory_snapshot.empty())
    return ToolResult::Error(-3, "No memory snapshot available — call snapshot_memory first");

  if (s_snapshot_base_address + s_snapshot_size > Bus::g_ram_size)
    return ToolResult::Error(-3, "Snapshot range is no longer valid (RAM size changed?)");

  static constexpr u32 MAX_CHANGES = 500;

  const u8* snap_ptr = s_memory_snapshot.data();
  const u8* live_ptr = Bus::g_ram + s_snapshot_base_address;

  u32 total_changes = 0;
  bool truncated = false;

  JsonWriter w;
  w.StartObject();
  w.KeyString("address", FormatHex32(0x80000000u | s_snapshot_base_address));
  w.KeyUint("size", s_snapshot_size);

  w.Key("changes");
  w.StartArray();

  for (u32 offset = 0; offset < s_snapshot_size; offset++)
  {
    if (snap_ptr[offset] == live_ptr[offset])
      continue;

    total_changes++;

    if (total_changes > MAX_CHANGES)
    {
      truncated = true;
      continue;
    }

    const u32 change_virt = 0x80000000u | (s_snapshot_base_address + offset);

    w.StartObject();
    w.KeyUint("offset", offset);
    w.KeyString("address", FormatHex32(change_virt));
    w.KeyUint("old_value", snap_ptr[offset]);
    w.KeyUint("new_value", live_ptr[offset]);
    w.EndObject();
  }

  w.EndArray();
  w.KeyUint("total_changes", total_changes);
  w.KeyBool("truncated", truncated);
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult HandleFindFreeRam(const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  u32 min_size = 256;
  if (args.contains("min_size") && args["min_size"].is_number_integer())
  {
    const u32 requested = static_cast<u32>(args["min_size"].get_int());
    if (requested > 0)
      min_size = requested;
  }

  u32 scan_start_phys = 0x00010000u;
  if (args.contains("start"))
  {
    const auto parsed = ParseAddress(args["start"]);
    if (!parsed.has_value())
      return ToolResult::Error(-32602, "Invalid 'start' parameter");
    scan_start_phys = parsed.value() & 0x1FFFFFFFu;
  }

  u32 scan_end_phys = Bus::g_ram_size;
  if (args.contains("end"))
  {
    const auto parsed = ParseAddress(args["end"]);
    if (!parsed.has_value())
      return ToolResult::Error(-32602, "Invalid 'end' parameter");
    scan_end_phys = parsed.value() & 0x1FFFFFFFu;
  }

  if (scan_start_phys >= scan_end_phys || scan_end_phys > Bus::g_ram_size)
    return ToolResult::Error(-32602, "Invalid scan range");

  static constexpr u32 MAX_REGIONS = 50;

  struct FreeRegion
  {
    u32 phys_start;
    u32 size;
  };

  std::vector<FreeRegion> regions;
  regions.reserve(64);

  const u8* ram = Bus::g_ram;
  u32 i = scan_start_phys;
  while (i < scan_end_phys)
  {
    if (ram[i] != 0)
    {
      i++;
      continue;
    }

    u32 run_start = i;
    while (i < scan_end_phys && ram[i] == 0)
      i++;

    const u32 run_size = i - run_start;
    if (run_size >= min_size)
      regions.push_back({run_start, run_size});
  }

  std::sort(regions.begin(), regions.end(),
            [](const FreeRegion& a, const FreeRegion& b) { return a.size > b.size; });

  const u32 reported_count = static_cast<u32>(std::min<size_t>(regions.size(), MAX_REGIONS));

  JsonWriter w;
  w.StartObject();
  w.KeyUint("min_size", min_size);
  w.KeyUint("count", reported_count);

  w.Key("regions");
  w.StartArray();
  for (u32 idx = 0; idx < reported_count; idx++)
  {
    const FreeRegion& r = regions[idx];
    const u32 virt_addr = 0x80000000u | r.phys_start;
    w.StartObject();
    w.KeyString("address", FormatHex32(virt_addr));
    w.KeyUint("size", r.size);
    w.EndObject();
  }
  w.EndArray();

  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

// ---- Tool dispatch table ----


// ============================================================
// Unified handler functions (Phase 2 consolidation)
// ============================================================

static ToolResult HandleGetStatusUnified(const JsonValue& args)
{
  const std::string detail = args.contains("detail") && args["detail"].is_string()
    ? std::string(args["detail"].get_string()) : "basic";
  if (detail == "full")
    return HandleGetEnhancedStatus(args);
  return HandleGetStatus(args);
}

static ToolResult HandleBreakpointUnified(const JsonValue& args)
{
  if (!args.contains("action") || !args["action"].is_string())
    return ToolResult::Error(-32602, "Missing 'action' parameter");
  const std::string action(args["action"].get_string());
  if (action == "list") return HandleListBreakpoints(args);
  if (action == "clear") return HandleClearBreakpoints(args);
  if (action == "add") return HandleAddBreakpoint(args);
  if (action == "remove") return HandleRemoveBreakpoint(args);
  if (action == "enable") return HandleEnableBreakpoint(args);
  if (action == "disable") return HandleDisableBreakpoint(args);
  return ToolResult::Error(-32602, "Invalid action. Use: add, remove, list, enable, disable, clear");
}

static ToolResult HandleGetGpuStateUnified(const JsonValue& args)
{
  const std::string aspect = args.contains("aspect") && args["aspect"].is_string()
    ? std::string(args["aspect"].get_string()) : "registers";
  if (aspect == "draw") return HandleGetGpuDrawState(args);
  if (aspect == "stats") return HandleGetGpuStats(args);
  if (aspect == "crtc") return HandleGetGpuCrtcState(args);
  if (aspect == "all") return HandleGetGpuState(args);
  return HandleGetGpuState(args); // "registers" or default
}

static ToolResult HandleGetSpuStateUnified(const JsonValue& args)
{
  const std::string detail = args.contains("detail") && args["detail"].is_string()
    ? std::string(args["detail"].get_string()) : "basic";
  if (detail == "voices") return HandleGetSpuVoiceState(args);
  if (detail == "reverb") return HandleGetSpuReverbState(args);
  return HandleGetSpuState(args);
}

static ToolResult HandleGetHardwareState(const JsonValue& args)
{
  if (!args.contains("subsystem") || !args["subsystem"].is_string())
    return ToolResult::Error(-32602, "Missing 'subsystem' parameter");
  const std::string subsystem(args["subsystem"].get_string());
  if (subsystem == "dma") return HandleGetDmaState(args);
  if (subsystem == "timers") return HandleGetTimersState(args);
  if (subsystem == "interrupts") return HandleGetInterruptState(args);
  if (subsystem == "mdec") return HandleGetMdecState(args);
  if (subsystem == "timing_events") return HandleGetTimingEvents(args);
  return ToolResult::Error(-32602, "Invalid subsystem. Use: dma, timers, interrupts, mdec, timing_events");
}

static ToolResult HandleMemoryScanUnified(const JsonValue& args)
{
  if (!args.contains("action") || !args["action"].is_string())
    return ToolResult::Error(-32602, "Missing 'action' parameter");
  const std::string action(args["action"].get_string());
  if (action == "start") return HandleMemoryScanStart(args);
  if (action == "refine") return HandleMemoryScanRefine(args);
  if (action == "results") return HandleMemoryScanResults(args);
  if (action == "reset") return HandleMemoryScanReset(args);
  return ToolResult::Error(-32602, "Invalid action. Use: start, refine, results, reset");
}

static ToolResult HandleMemoryWatchUnified(const JsonValue& args)
{
  if (!args.contains("action") || !args["action"].is_string())
    return ToolResult::Error(-32602, "Missing 'action' parameter");
  const std::string action(args["action"].get_string());
  if (action == "add") return HandleAddMemoryWatch(args);
  if (action == "remove") return HandleRemoveMemoryWatch(args);
  if (action == "list") return HandleListMemoryWatches(args);
  return ToolResult::Error(-32602, "Invalid action. Use: add, remove, list");
}

static ToolResult HandleVramWatchUnified(const JsonValue& args)
{
  if (!args.contains("action") || !args["action"].is_string())
    return ToolResult::Error(-32602, "Missing 'action' parameter");
  const std::string action(args["action"].get_string());
  if (action == "add") return HandleWatchVramWrite(args);
  if (action == "remove") return HandleRemoveVramWatch(args);
  if (action == "list") return HandleListVramWatches(args);
  if (action == "last_hit") return HandleGetVramWatchLastHit(args);
  return ToolResult::Error(-32602, "Invalid action. Use: add, remove, list, last_hit");
}

static ToolResult HandleCaptureUnified(const JsonValue& args)
{
  if (!args.contains("action") || !args["action"].is_string())
    return ToolResult::Error(-32602, "Missing 'action' parameter");
  const std::string action(args["action"].get_string());
  if (action == "start") return HandleStartCapture(args);
  if (action == "stop") return HandleStopCapture(args);
  if (action == "status") return HandleGetCaptureStatus(args);
  return ToolResult::Error(-32602, "Invalid action. Use: start, stop, status");
}

static ToolResult HandleTraceUnified(const JsonValue& args)
{
  if (!args.contains("action") || !args["action"].is_string())
    return ToolResult::Error(-32602, "Missing 'action' parameter");
  const std::string action(args["action"].get_string());
  if (action == "start") return HandleStartTrace(args);
  if (action == "stop") return HandleStopTrace(args);
  if (action == "status") return HandleGetTraceStatus(args);
  return ToolResult::Error(-32602, "Invalid action. Use: start, stop, status");
}

static ToolResult HandleGpuDumpUnified(const JsonValue& args)
{
  if (!args.contains("action") || !args["action"].is_string())
    return ToolResult::Error(-32602, "Missing 'action' parameter");
  const std::string action(args["action"].get_string());
  if (action == "start") return HandleStartGpuDump(args);
  if (action == "stop") return HandleStopGpuDump(args);
  return ToolResult::Error(-32602, "Invalid action. Use: start, stop");
}

static ToolResult HandleGetCdromStateUnified(const JsonValue& args)
{
  const std::string detail = args.contains("detail") && args["detail"].is_string()
    ? std::string(args["detail"].get_string()) : "basic";
  if (detail == "extended") return HandleGetCdromExtendedState(args);
  return HandleGetCdromState(args);
}

static ToolResult HandleGetAchievementsStateUnified(const JsonValue& args)
{
  const std::string detail = args.contains("detail") && args["detail"].is_string()
    ? std::string(args["detail"].get_string()) : "basic";
  if (detail == "full") return HandleGetAchievementsDetails(args);
  return HandleGetAchievementsState(args);
}

static ToolResult HandleGetMemoryCardInfoUnified(const JsonValue& args)
{
  // Make slot optional with default 0
  if (!args.contains("slot") || !args["slot"].is_number())
    return HandleGetMemoryCardInfo(*JsonValue::Parse(R"({"slot":0})"));
  return HandleGetMemoryCardInfo(args);
}

static ToolResult HandleGetGameInfoUnified(const JsonValue& args)
{
  // If serial or path provided, use existing handler
  // If neither provided, falls through to current game info + database info + play time
  return HandleGetGameInfo(args);
}

static ToolResult HandleSetSpeedUnified(const JsonValue& args)
{
  return HandleSetSpeed(args);
}

static ToolResult HandleDiscControlUnified(const JsonValue& args)
{
  if (!args.contains("action") || !args["action"].is_string())
    return ToolResult::Error(-32602, "Missing 'action' parameter");
  const std::string action(args["action"].get_string());
  if (action == "insert") return HandleInsertDisc(args);
  if (action == "eject") return HandleEjectDisc(args);
  if (action == "switch") return HandleSwitchDisc(args);
  if (action == "list") return HandleListDiscs(args);
  return ToolResult::Error(-32602, "Invalid action. Use: insert, eject, switch, list");
}

static ToolResult HandleWaitForPause([[maybe_unused]] const JsonValue& args)
{
  if (!System::IsValid())
    return ToolResult::Error(-1, "System not running");

  JsonWriter w;
  w.StartObject();
  if (System::IsPaused())
  {
    w.KeyString("status", "paused");
    w.KeyString("pc", FormatHex32(CPU::g_state.pc));
  }
  else
  {
    w.KeyString("status", "running");
    w.KeyString("message", "System is still running. Poll again to check if it has paused.");
  }
  w.EndObject();
  return ToolResult{w.TakeOutput()};
}

static ToolResult DispatchToolCall(const std::string& tool_name, const JsonValue& args)
{
  if (tool_name == "read_registers")
    return HandleReadRegisters(args);
  else if (tool_name == "write_register")
    return HandleWriteRegister(args);
  else if (tool_name == "disassemble")
    return HandleDisassemble(args);
  else if (tool_name == "step_into")
    return HandleStepInto(args);
  else if (tool_name == "step_over")
    return HandleStepOver(args);
  else if (tool_name == "step_out")
    return HandleStepOut(args);
  else if (tool_name == "pause")
    return HandlePause(args);
  else if (tool_name == "continue")
    return HandleContinue(args);
  else if (tool_name == "read_memory")
    return HandleReadMemory(args);
  else if (tool_name == "write_memory")
    return HandleWriteMemory(args);
  else if (tool_name == "search_memory")
    return HandleSearchMemory(args);
  else if (tool_name == "dump_ram")
    return HandleDumpRam(args);
  else if (tool_name == "get_gpu_state")
    return HandleGetGpuStateUnified(args);
  else if (tool_name == "get_spu_state")
    return HandleGetSpuStateUnified(args);
  else if (tool_name == "get_cdrom_state")
    return HandleGetCdromStateUnified(args);
  else if (tool_name == "dump_vram")
    return HandleDumpVram(args);
  else if (tool_name == "dump_spu_ram")
    return HandleDumpSpuRam(args);
  else if (tool_name == "get_status")
    return HandleGetStatusUnified(args);
  else if (tool_name == "reset")
    return HandleReset(args);
  else if (tool_name == "save_state")
    return HandleSaveState(args);
  else if (tool_name == "load_state")
    return HandleLoadState(args);
  else if (tool_name == "frame_step")
    return HandleFrameStep(args);
  // Controller input
  else if (tool_name == "press_button")
    return HandlePressButton(args);
  else if (tool_name == "release_button")
    return HandleReleaseButton(args);
  else if (tool_name == "set_analog")
    return HandleSetAnalog(args);
  else if (tool_name == "get_controller_state")
    return HandleGetControllerState(args);
  else if (tool_name == "list_controllers")
    return HandleListControllers(args);
  else if (tool_name == "input_sequence")
    return HandleInputSequence(args);
  // Settings
  else if (tool_name == "get_settings")
    return HandleGetSettings(args);
  else if (tool_name == "set_setting")
    return HandleSetSetting(args);
  else if (tool_name == "set_speed")
    return HandleSetSpeedUnified(args);
  else if (tool_name == "take_screenshot")
    return HandleTakeScreenshot(args);
  // Cheats
  else if (tool_name == "list_cheats")
    return HandleListCheats(args);
  else if (tool_name == "toggle_cheat")
    return HandleToggleCheat(args);
  else if (tool_name == "apply_cheat")
    return HandleApplyCheat(args);
  else if (tool_name == "get_cheat_status")
    return HandleGetCheatStatus(args);
  // Media management
  else if (tool_name == "list_save_states")
    return HandleListSaveStates(args);
  else if (tool_name == "swap_memory_cards")
    return HandleSwapMemoryCards(args);
  else if (tool_name == "boot_game")
    return HandleBootGame(args);
  else if (tool_name == "shutdown_system")
    return HandleShutdownSystem(args);
  // CPU debug enhancement tools
  else if (tool_name == "get_cop0_state")
    return HandleGetCop0State(args);
  else if (tool_name == "get_cpu_execution_state")
    return HandleGetCpuExecutionState(args);
  // GTE tools
  else if (tool_name == "get_gte_registers")
    return HandleGetGteRegisters(args);
  else if (tool_name == "set_gte_register")
    return HandleSetGteRegister(args);
  // Hardware debug tools
  else if (tool_name == "get_memory_map")
    return HandleGetMemoryMap(args);
  // System/game tools
  else if (tool_name == "inject_executable")
    return HandleInjectExecutable(args);
  else if (tool_name == "get_achievements_state")
    return HandleGetAchievementsStateUnified(args);
  // Advanced state tools
  else if (tool_name == "get_cpu_icache_state")
    return HandleGetCpuIcacheState(args);
  else if (tool_name == "get_pgxp_state")
    return HandleGetPgxpState(args);
  // Game list tools
  else if (tool_name == "list_games")
    return HandleListGames(args);
  else if (tool_name == "get_game_info")
    return HandleGetGameInfoUnified(args);
  else if (tool_name == "refresh_game_list")
    return HandleRefreshGameList(args);
  // BIOS tools
  else if (tool_name == "list_bios")
    return HandleListBios(args);
  else if (tool_name == "get_bios_info")
    return HandleGetBiosInfo(args);
  // Save state extended tools
  else if (tool_name == "get_save_state_info")
    return HandleGetSaveStateInfo(args);
  else if (tool_name == "delete_save_states")
    return HandleDeleteSaveStates(args);
  else if (tool_name == "undo_load_state")
    return HandleUndoLoadState(args);
  else if (tool_name == "get_save_state_screenshot")
    return HandleGetSaveStateScreenshot(args);
  // Cheats advanced tools
  else if (tool_name == "get_cheat_details")
    return HandleGetCheatDetails(args);
  else if (tool_name == "create_cheat")
    return HandleCreateCheat(args);
  else if (tool_name == "import_cheats")
    return HandleImportCheats(args);
  else if (tool_name == "export_cheats")
    return HandleExportCheats(args);
  else if (tool_name == "validate_cheat")
    return HandleValidateCheat(args);
  // Memory card advanced tools
  else if (tool_name == "get_memory_card_info")
    return HandleGetMemoryCardInfoUnified(args);
  else if (tool_name == "read_memory_card_file")
    return HandleReadMemoryCardFile(args);
  else if (tool_name == "delete_memory_card_file")
    return HandleDeleteMemoryCardFile(args);
  else if (tool_name == "export_memory_card_save")
    return HandleExportMemoryCardSave(args);
  else if (tool_name == "import_memory_card_save")
    return HandleImportMemoryCardSave(args);
  // Post-processing tools
  else if (tool_name == "list_shaders")
    return HandleListShaders(args);
  else if (tool_name == "get_shader_chain")
    return HandleGetShaderChain(args);
  else if (tool_name == "add_shader")
    return HandleAddShader(args);
  else if (tool_name == "remove_shader")
    return HandleRemoveShader(args);
  else if (tool_name == "get_shader_options")
    return HandleGetShaderOptions(args);
  else if (tool_name == "set_shader_option")
    return HandleSetShaderOption(args);
  // Media capture tools
  // Memory scanner tools
  // Hotkeys & system tools
  else if (tool_name == "list_hotkeys")
    return HandleListHotkeys(args);
  else if (tool_name == "trigger_hotkey")
    return HandleTriggerHotkey(args);
  // Achievements & disc info tools
  else if (tool_name == "get_disc_info")
    return HandleGetDiscInfo(args);
  // ROM hacking tools
  else if (tool_name == "list_disc_files")
    return HandleListDiscFiles(args);
  else if (tool_name == "read_disc_file")
    return HandleReadDiscFile(args);
  else if (tool_name == "read_disc_sectors")
    return HandleReadDiscSectors(args);
  else if (tool_name == "read_vram_region")
    return HandleReadVramRegion(args);
  else if (tool_name == "write_vram_region")
    return HandleWriteVramRegion(args);
  else if (tool_name == "snapshot_memory")
    return HandleSnapshotMemory(args);
  else if (tool_name == "diff_memory")
    return HandleDiffMemory(args);
  else if (tool_name == "find_free_ram")
    return HandleFindFreeRam(args);

  else if (tool_name == "breakpoint")
    return HandleBreakpointUnified(args);
  else if (tool_name == "get_hardware_state")
    return HandleGetHardwareState(args);
  else if (tool_name == "memory_scan")
    return HandleMemoryScanUnified(args);
  else if (tool_name == "memory_watch")
    return HandleMemoryWatchUnified(args);
  else if (tool_name == "vram_watch")
    return HandleVramWatchUnified(args);
  else if (tool_name == "capture")
    return HandleCaptureUnified(args);
  else if (tool_name == "trace")
    return HandleTraceUnified(args);
  else if (tool_name == "gpu_dump")
    return HandleGpuDumpUnified(args);
  else if (tool_name == "disc_control")
    return HandleDiscControlUnified(args);
  else if (tool_name == "wait_for_pause")
    return HandleWaitForPause(args);

  return ToolResult::Error(-32601, fmt::format("Unknown tool: {}", tool_name));
}

static void LogStreamCallback(void* /*pUserParam*/, Log::MessageCategory category,
                               const char* /*functionName*/, std::string_view message)
{
  const Log::Level level = Log::UnpackLevel(category);
  if (level > s_log_stream_level)
    return;

  const Log::Channel channel = Log::UnpackChannel(category);
  const char* level_name = "info";
  switch (level)
  {
    case Log::Level::Error:
      level_name = "error";
      break;
    case Log::Level::Warning:
      level_name = "warning";
      break;
    case Log::Level::Info:
      level_name = "info";
      break;
    default:
      level_name = "debug";
      break;
  }

  std::string escaped_message;
  escaped_message += '"';
  JsonWriter::EscapeString(escaped_message, message);
  escaped_message += '"';
  const std::string data = fmt::format("{{\"level\":\"{}\",\"logger\":\"{}\",\"data\":{}}}",
                                        level_name, Log::GetChannelName(channel),
                                        escaped_message);
  BroadcastSSEEvent("notifications/message", data);
}

static void NotifyResourceUpdated(const std::string& uri)
{
  if (std::find(s_subscribed_resources.begin(), s_subscribed_resources.end(), uri) !=
      s_subscribed_resources.end())
  {
    const std::string data = fmt::format("{{\"uri\":\"{}\"}}", uri);
    BroadcastSSEEvent("notifications/resources/updated", data);
  }
}

} // namespace MCPServer

MCPServer::ClientSocket::ClientSocket(SocketMultiplexer& multiplexer, SocketDescriptor descriptor)
  : BufferedStreamSocket(multiplexer, descriptor, 65536, 65536)
{
}

MCPServer::ClientSocket::~ClientSocket() = default;

void MCPServer::ClientSocket::OnConnected()
{
  INFO_LOG("MCP client {} connected.", GetRemoteAddress().ToString());
  s_mcp_clients.push_back(std::static_pointer_cast<ClientSocket>(shared_from_this()));
}

void MCPServer::ClientSocket::OnDisconnected(const Error& error)
{
  INFO_LOG("MCP client {} disconnected: {}", GetRemoteAddress().ToString(), error.GetDescription());

  // Remove from SSE clients if present.
  {
    const auto sse_iter =
      std::find_if(s_sse_clients.begin(), s_sse_clients.end(),
                   [this](const std::shared_ptr<ClientSocket>& rhs) { return (rhs.get() == this); });
    if (sse_iter != s_sse_clients.end())
      s_sse_clients.erase(sse_iter);
  }

  // Remove from main client list.
  const auto iter =
    std::find_if(s_mcp_clients.begin(), s_mcp_clients.end(),
                 [this](const std::shared_ptr<ClientSocket>& rhs) { return (rhs.get() == this); });
  if (iter == s_mcp_clients.end())
  {
    ERROR_LOG("Unknown MCP client disconnected? This should never happen.");
    return;
  }

  s_mcp_clients.erase(iter);
}

void MCPServer::ClientSocket::OnRead()
{
  const std::span<const u8> buffer = AcquireReadBuffer();
  if (buffer.empty())
    return;

  // Append incoming data to our accumulation buffer.
  m_recv_buffer.append(reinterpret_cast<const char*>(buffer.data()), buffer.size());
  ReleaseReadBuffer(buffer.size());

  // Try to parse a complete HTTP request.
  for (;;)
  {
    // Look for end of headers.
    const size_t header_end = m_recv_buffer.find("\r\n\r\n");
    if (header_end == std::string::npos)
      return; // Need more data.

    const size_t headers_len = header_end + 4; // Include the \r\n\r\n.
    const std::string_view headers_view(m_recv_buffer.data(), header_end);

    // Parse request line.
    const size_t request_line_end = headers_view.find("\r\n");
    if (request_line_end == std::string_view::npos)
    {
      ERROR_LOG("Malformed HTTP request from {}", GetRemoteAddress().ToString());
      m_recv_buffer.clear();
      return;
    }

    const std::string_view request_line = headers_view.substr(0, request_line_end);

    // Parse METHOD PATH HTTP/1.x
    const size_t first_space = request_line.find(' ');
    const size_t second_space = (first_space != std::string_view::npos)
                                  ? request_line.find(' ', first_space + 1)
                                  : std::string_view::npos;
    if (first_space == std::string_view::npos || second_space == std::string_view::npos)
    {
      ERROR_LOG("Malformed HTTP request line from {}: {}", GetRemoteAddress().ToString(), request_line);
      m_recv_buffer.clear();
      return;
    }

    const std::string method(request_line.substr(0, first_space));
    const std::string path(request_line.substr(first_space + 1, second_space - first_space - 1));

    // Parse Content-Length header for POST requests.
    size_t content_length = 0;
    if (method == "POST")
    {
      const std::string_view remaining_headers = headers_view.substr(request_line_end + 2);
      size_t pos = 0;
      while (pos < remaining_headers.size())
      {
        const size_t line_end = remaining_headers.find("\r\n", pos);
        const std::string_view line = (line_end != std::string_view::npos)
                                        ? remaining_headers.substr(pos, line_end - pos)
                                        : remaining_headers.substr(pos);

        // Case-insensitive check for Content-Length.
        const size_t colon_pos = line.find(':');
        if (colon_pos != std::string_view::npos)
        {
          const std::string_view header_name = line.substr(0, colon_pos);
          if (StringUtil::EqualNoCase(header_name, "content-length"))
          {
            std::string_view value = line.substr(colon_pos + 1);
            // Trim leading whitespace from the value.
            while (!value.empty() && value[0] == ' ')
              value.remove_prefix(1);
            const std::optional<u32> val = StringUtil::FromChars<u32>(value);
            if (val.has_value())
              content_length = static_cast<size_t>(val.value());
          }
        }

        if (line_end == std::string_view::npos)
          break;
        pos = line_end + 2;
      }

      // Reject oversized bodies.
      if (content_length > MAX_HTTP_BODY_SIZE)
      {
        ERROR_LOG("HTTP body too large ({} bytes) from {}", content_length,
                  GetRemoteAddress().ToString());
        SendHttpResponse(413, "text/plain", "Request body too large");
        m_recv_buffer.clear();
        return;
      }

      // Wait for the full body.
      if (m_recv_buffer.size() < headers_len + content_length)
        return; // Need more data.
    }

    // Extract the body (if any).
    const std::string body = m_recv_buffer.substr(headers_len, content_length);

    // Parse relevant HTTP headers for Streamable HTTP transport.
    m_accept_sse = false;
    m_origin_header.clear();
    m_mcp_session_id_header.clear();
    m_mcp_protocol_version_header.clear();
    {
      const std::string_view remaining_headers = headers_view.substr(request_line_end + 2);
      size_t pos = 0;
      while (pos < remaining_headers.size())
      {
        const size_t line_end = remaining_headers.find("\r\n", pos);
        const std::string_view line = (line_end != std::string_view::npos)
                                        ? remaining_headers.substr(pos, line_end - pos)
                                        : remaining_headers.substr(pos);
        const size_t colon_pos = line.find(':');
        if (colon_pos != std::string_view::npos)
        {
          const std::string_view header_name = line.substr(0, colon_pos);
          std::string_view value = line.substr(colon_pos + 1);
          while (!value.empty() && value.front() == ' ')
            value.remove_prefix(1);

          if (StringUtil::EqualNoCase(header_name, "accept"))
            m_accept_sse = (value.find("text/event-stream") != std::string_view::npos);
          else if (StringUtil::EqualNoCase(header_name, "origin"))
            m_origin_header = std::string(value);
          else if (StringUtil::EqualNoCase(header_name, "mcp-session-id"))
            m_mcp_session_id_header = std::string(value);
          else if (StringUtil::EqualNoCase(header_name, "mcp-protocol-version"))
            m_mcp_protocol_version_header = std::string(value);
        }
        if (line_end == std::string_view::npos)
          break;
        pos = line_end + 2;
      }
    }

    // Check Authorization header if auth is required.
    if (!s_auth_token.empty())
    {
      m_last_auth_ok = false;
      const std::string expected = fmt::format("Bearer {}", s_auth_token);
      // Search for "Authorization:" header (case-insensitive).
      const std::string headers_str = m_recv_buffer.substr(0, headers_len);
      size_t auth_pos = std::string::npos;
      for (size_t i = 0; i + 14 <= headers_len; ++i)
      {
        if (StringUtil::Strncasecmp(headers_str.c_str() + i, "authorization:", 14) == 0)
        {
          auth_pos = i + 14;
          break;
        }
      }
      if (auth_pos != std::string::npos)
      {
        const size_t line_end = headers_str.find("\r\n", auth_pos);
        std::string_view value(headers_str.c_str() + auth_pos,
                               (line_end != std::string::npos ? line_end : headers_len) - auth_pos);
        while (!value.empty() && value.front() == ' ')
          value.remove_prefix(1);
        m_last_auth_ok = (value == expected);
      }
    }
    else
    {
      m_last_auth_ok = true; // No auth required.
    }

    // Remove the processed request from the buffer.
    m_recv_buffer.erase(0, headers_len + content_length);

    // Handle CORS preflight requests.
    if (method == "OPTIONS")
    {
      const std::string response = fmt::format("HTTP/1.1 204 No Content\r\n"
                                               "Access-Control-Allow-Origin: {}\r\n"
                                               "Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\n"
                                               "Access-Control-Allow-Headers: Content-Type, Authorization, MCP-Session-Id, MCP-Protocol-Version\r\n"
                                               "Access-Control-Expose-Headers: MCP-Session-Id\r\n"
                                               "Connection: keep-alive\r\n"
                                               "Content-Length: 0\r\n"
                                               "\r\n",
                                               s_cors_origin);
      Write(response.data(), response.size());
      continue;
    }

    // Process the request.
    ProcessHttpRequest(method, path, body);
  }
}

void MCPServer::ClientSocket::ProcessHttpRequest(const std::string& method, const std::string& path,
                                                 const std::string& body)
{
  DEV_LOG("HTTP {} {} from {}", method, path, GetRemoteAddress().ToString());

  // Origin validation (MCP spec §2.2 Security Warning): MUST validate Origin header.
  // If Origin is present and doesn't match the allowed origin, reject with 403.
  if (!m_origin_header.empty() && s_cors_origin != "*")
  {
    if (m_origin_header != s_cors_origin)
    {
      WARNING_LOG("Rejected request with invalid Origin '{}' from {}", m_origin_header,
                  GetRemoteAddress().ToString());
      SendHttpResponseDirect(403, "text/plain", "Forbidden: invalid Origin");
      return;
    }
  }

  // Authentication check.
  if (!m_last_auth_ok && method != "OPTIONS")
  {
    WARNING_LOG("Unauthorized request from {}", GetRemoteAddress().ToString());
    SendHttpResponseDirect(401, "text/plain", "Unauthorized: missing or invalid Bearer token");
    return;
  }

  // Extract base path (before query string).
  const std::string base_path = path.substr(0, path.find('?'));

  if (method == "POST" && base_path == "/mcp")
  {
    // MCP-Session-Id validation: if a session exists and this is not an initialize request,
    // the client MUST include the session ID. We check after JSON parse in ProcessJsonRpc
    // for initialize detection, but validate the header presence here for non-initialize.
    // (Deferred to ProcessJsonRpc where we know the method.)

    // MCP-Protocol-Version validation: if present, must match negotiated version.
    if (!m_mcp_protocol_version_header.empty() && !s_negotiated_protocol_version.empty())
    {
      if (m_mcp_protocol_version_header != s_negotiated_protocol_version)
      {
        WARNING_LOG("Invalid MCP-Protocol-Version '{}' (expected '{}') from {}",
                    m_mcp_protocol_version_header, s_negotiated_protocol_version,
                    GetRemoteAddress().ToString());
        SendHttpResponseDirect(400, "application/json",
          MakeJsonRpcError(JsonValue(), -32600,
            fmt::format("Invalid MCP-Protocol-Version (expected '{}')", s_negotiated_protocol_version)));
        return;
      }
    }

    if (m_accept_sse)
    {
      // Streamable HTTP transport (MCP spec 2025-11-25 §2.1):
      // Client sent Accept: text/event-stream on POST — respond with SSE stream.
      if (s_sse_clients.size() >= MAX_SSE_CLIENTS)
      {
        WARNING_LOG("Maximum SSE clients ({}) reached, rejecting from {}",
                    MAX_SSE_CLIENTS, GetRemoteAddress().ToString());
        SendHttpResponseDirect(503, "text/plain", "Too many SSE connections");
        return;
      }

      // Register this connection for server-initiated events.
      m_is_sse = true;
      s_sse_clients.push_back(std::static_pointer_cast<ClientSocket>(shared_from_this()));

      // Send SSE response headers with MCP-Session-Id (if session exists).
      std::string extra_headers;
      if (!s_mcp_session_id.empty())
        extra_headers = fmt::format("MCP-Session-Id: {}\r\n", s_mcp_session_id);

      const std::string sse_headers =
        fmt::format("HTTP/1.1 200 OK\r\n"
                    "Content-Type: text/event-stream\r\n"
                    "Cache-Control: no-cache\r\n"
                    "Connection: keep-alive\r\n"
                    "{}Access-Control-Allow-Origin: {}\r\n"
                    "\r\n",
                    extra_headers, s_cors_origin);
      Write(sse_headers.data(), sse_headers.size());

      // Process the JSON-RPC request; response goes as SSE event on this connection.
      m_respond_via_sse = true;
      ProcessJsonRpc(body);
      m_respond_via_sse = false;

      // Per spec §2.1.6: after response sent, server SHOULD terminate the SSE stream.
      // Remove from SSE clients so this connection can be reused for future POSTs.
      m_is_sse = false;
      auto it = std::find(s_sse_clients.begin(), s_sse_clients.end(),
                          std::static_pointer_cast<ClientSocket>(shared_from_this()));
      if (it != s_sse_clients.end())
        s_sse_clients.erase(it);
    }
    else
    {
      // Plain HTTP POST — direct JSON response (no SSE).
      ProcessJsonRpc(body);
    }
  }
  else if (method == "GET" && base_path == "/mcp")
  {
    // Streamable HTTP §2.2: client MAY issue GET to open SSE stream for server-initiated messages.
    if (!m_accept_sse)
    {
      SendHttpResponseDirect(405, "text/plain", "Method Not Allowed");
      return;
    }

    // Session validation: GET requires a valid session.
    if (!s_mcp_session_id.empty() && m_mcp_session_id_header != s_mcp_session_id)
    {
      SendHttpResponseDirect(400, "text/plain", "Bad Request: missing or invalid MCP-Session-Id");
      return;
    }

    if (m_is_sse)
    {
      SendHttpResponseDirect(400, "text/plain", "Already in SSE mode");
      return;
    }

    if (s_sse_clients.size() >= MAX_SSE_CLIENTS)
    {
      SendHttpResponseDirect(503, "text/plain", "Too many SSE connections");
      return;
    }

    // Open a long-lived SSE stream for server-initiated notifications/requests.
    m_is_sse = true;
    s_sse_clients.push_back(std::static_pointer_cast<ClientSocket>(shared_from_this()));

    INFO_LOG("Client {} opened GET SSE stream for server events.", GetRemoteAddress().ToString());

    std::string extra_headers;
    if (!s_mcp_session_id.empty())
      extra_headers = fmt::format("MCP-Session-Id: {}\r\n", s_mcp_session_id);

    const std::string sse_headers =
      fmt::format("HTTP/1.1 200 OK\r\n"
                  "Content-Type: text/event-stream\r\n"
                  "Cache-Control: no-cache\r\n"
                  "Connection: keep-alive\r\n"
                  "{}Access-Control-Allow-Origin: {}\r\n"
                  "\r\n",
                  extra_headers, s_cors_origin);
    Write(sse_headers.data(), sse_headers.size());
    // Connection stays open — server sends events via BroadcastSSEEvent / OnSystemPaused / etc.
  }
  else if (method == "DELETE" && base_path == "/mcp")
  {
    // Streamable HTTP §2.5.5: session termination via DELETE with MCP-Session-Id header.
    if (m_mcp_session_id_header.empty())
    {
      SendHttpResponseDirect(400, "text/plain", "Bad Request: missing MCP-Session-Id header");
      return;
    }

    if (s_mcp_session_id.empty() || m_mcp_session_id_header != s_mcp_session_id)
    {
      SendHttpResponseDirect(404, "text/plain", "Session not found");
      return;
    }

    INFO_LOG("Session {} terminated by DELETE from {}", s_mcp_session_id,
             GetRemoteAddress().ToString());

    // Terminate session: clear session ID and disconnect all SSE clients.
    s_mcp_session_id.clear();
    s_negotiated_protocol_version.clear();
    for (auto& client : s_sse_clients)
      client->m_is_sse = false;
    s_sse_clients.clear();

    SendHttpResponseDirect(204, "text/plain", "");
  }
  else
  {
    WARNING_LOG("404 for {} {} from {}", method, path, GetRemoteAddress().ToString());
    SendHttpResponseDirect(404, "text/plain", "Not Found");
  }
}

void MCPServer::ClientSocket::ProcessJsonRpc(const std::string& json_body)
{
  std::optional<JsonValue> parsed = JsonValue::Parse(json_body);
  if (!parsed.has_value())
  {
    ERROR_LOG("Failed to parse JSON-RPC request from {}", GetRemoteAddress().ToString());
    const std::string error_response = MakeJsonRpcError(JsonValue(), -32700, "Parse error");
    SendHttpResponse(200, "application/json", error_response);
    return;
  }
  const JsonValue& request = parsed.value();

  // Validate jsonrpc field per JSON-RPC 2.0 spec (section 4.1).
  const JsonValue& jsonrpc_val = request["jsonrpc"];
  if (jsonrpc_val.is_null() || !jsonrpc_val.is_string() ||
      std::string_view(jsonrpc_val.get_string()) != "2.0")
  {
    ERROR_LOG("JSON-RPC request missing or invalid 'jsonrpc' field");
    const std::string error_response =
      MakeJsonRpcError(JsonValue(), -32600, "Invalid Request: missing or invalid 'jsonrpc' field (must be \"2.0\")");
    SendHttpResponse(200, "application/json", error_response);
    return;
  }

  const JsonValue& method_val = request["method"];
  if (method_val.is_null() || !method_val.is_string())
  {
    ERROR_LOG("JSON-RPC request missing 'method' field");
    const std::string error_response = MakeJsonRpcError(JsonValue(), -32600, "Invalid Request");
    SendHttpResponse(200, "application/json", error_response);
    return;
  }
  const std::string method(method_val.get_string());

  const JsonValue& id = request["id"]; // returns NULL_VALUE if missing
  const bool is_notification = id.is_null() && !request.contains("id");
  const bool is_response = !request.contains("method");

  DEV_LOG("JSON-RPC method='{}' notification={}", method, is_notification);

  // MCP spec §2.1.4: notifications and responses → 202 Accepted with no body.
  if (is_notification || is_response)
  {
    // Handle specific notifications we care about.
    if (method == "notifications/initialized")
      DEV_LOG("MCP client initialized notification received.");

    SendHttpResponse(202, "text/plain", "");
    return;
  }

  // MCP-Session-Id validation: if session exists, non-initialize requests MUST include it.
  if (!s_mcp_session_id.empty() && method != "initialize")
  {
    if (m_mcp_session_id_header != s_mcp_session_id)
    {
      WARNING_LOG("Request missing or invalid MCP-Session-Id from {}", GetRemoteAddress().ToString());
      const std::string error = MakeJsonRpcError(id, -32600, "Bad Request: missing or invalid MCP-Session-Id");
      SendHttpResponse(400, "application/json", error);
      return;
    }
  }

  if (method == "initialize")
  {
    // Read client's requested protocol version.
    const JsonValue& params = request["params"];
    const std::string client_version = (params.is_object() && params.contains("protocolVersion") &&
                                        params["protocolVersion"].is_string())
      ? std::string(params["protocolVersion"].get_string())
      : std::string();

    // Supported versions (newest first).
    static constexpr const char* SUPPORTED_VERSIONS[] = {"2025-11-25"};
    std::string negotiated_version = SUPPORTED_VERSIONS[0];

    if (!client_version.empty())
    {
      bool found = false;
      for (const char* v : SUPPORTED_VERSIONS)
      {
        if (client_version == v)
        {
          negotiated_version = v;
          found = true;
          break;
        }
      }
      if (!found)
        WARNING_LOG("Client requested unsupported protocol version '{}', using '{}'",
                    client_version, SUPPORTED_VERSIONS[0]);
    }

    // Generate a new crypto-secure session ID.
    s_mcp_session_id = GenerateSessionId();
    s_negotiated_protocol_version = negotiated_version;
    INFO_LOG("MCP session initialized: id={}, protocol={}", s_mcp_session_id, negotiated_version);

    JsonWriter w;
    w.StartObject();
    w.KeyString("protocolVersion", negotiated_version);
    w.Key("capabilities");
    w.StartObject();
    w.Key("tools"); w.StartObject(); w.KeyBool("listChanged", false); w.EndObject();
    w.Key("resources"); w.StartObject(); w.KeyBool("subscribe", true); w.KeyBool("listChanged", false); w.EndObject();
    w.Key("prompts"); w.StartObject(); w.KeyBool("listChanged", false); w.EndObject();
    w.Key("logging"); w.StartObject(); w.EndObject();
    w.EndObject();
    w.Key("serverInfo");
    w.StartObject();
    w.KeyString("name", "duckstation-mcp");
    w.KeyString("version", "1.0.0");
    w.EndObject();
    w.EndObject();
    const std::string response = MakeJsonRpcResponse(id, w.GetOutput());
    // Include MCP-Session-Id header in the InitializeResult response.
    SendHttpResponseWithSessionId(200, "application/json", response);
  }
  else if (method == "ping")
  {
    const std::string response = MakeJsonRpcResponse(id, "{}");
    SendHttpResponse(200, "application/json", response);
  }
  else if (method == "tools/list")
  {
    // clang-format off
    // Static tool definitions as raw JSON.
    static constexpr const char TOOLS_JSON[] = R"json({"tools":[
{"name":"read_registers","description":"Read CPU registers (GPR, COP0, GTE). Returns: JSON object with register names as keys and hex string values (e.g. {\"v0\":\"0x00000001\", \"sp\":\"0x801FFF00\", ...}).","inputSchema":{"type":"object","properties":{"group":{"type":"string","enum":["gpr","cop0","gte","all"],"default":"all","description":"Register group to read"}}},"title":"Read Registers","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"write_register","description":"Write a value to a CPU register by name. Returns: {register, value} confirming the written value.","inputSchema":{"type":"object","properties":{"name":{"type":"string","description":"Register name (e.g. 'v0', 'sp', 'pc', 'COP0_SR')"},"value":{"description":"Value to write (integer or hex string like '0x80010000')"}},"required":["name","value"]},"title":"Write Register","annotations":{"readOnlyHint":false,"destructiveHint":true,"idempotentHint":false,"openWorldHint":false}},
{"name":"disassemble","description":"Disassemble MIPS instructions at a given address. Returns: JSON array of {address, bytes, instruction, comment} objects, one per instruction.","inputSchema":{"type":"object","properties":{"address":{"description":"Start address (integer or hex string)"},"count":{"type":"integer","default":20,"description":"Number of instructions to disassemble"}},"required":["address"]},"title":"Disassemble","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"step_into","description":"Execute a single CPU instruction and return register state. Returns: GPR register snapshot (same format as read_registers with group=gpr).","inputSchema":{"type":"object","properties":{}},"title":"Step Into","annotations":{"readOnlyHint":false,"destructiveHint":false,"idempotentHint":false,"openWorldHint":false}},
{"name":"step_over","description":"Step over the current instruction. WARNING: This RESUMES execution and returns immediately. The system will pause when the next instruction is reached or after the call returns. Unlike step_into (synchronous), step_over is asynchronous. Poll get_status to detect when paused. Returns: {status} confirmation message.","inputSchema":{"type":"object","properties":{}},"title":"Step Over","annotations":{"readOnlyHint":false,"destructiveHint":false,"idempotentHint":false,"openWorldHint":false}},
{"name":"step_out","description":"Run until the current function returns. WARNING: This RESUMES execution and returns immediately (asynchronous). Poll get_status to detect when the system pauses. Returns: {status} confirmation message.","inputSchema":{"type":"object","properties":{"max_instructions":{"type":"integer","default":10000,"description":"Maximum instructions to search for return"}}},"title":"Step Out","annotations":{"readOnlyHint":false,"destructiveHint":false,"idempotentHint":false,"openWorldHint":false}},
{"name":"pause","description":"Pause the running system","inputSchema":{"type":"object","properties":{}},"title":"Pause","annotations":{"readOnlyHint":false,"destructiveHint":false,"idempotentHint":false,"openWorldHint":false}},
{"name":"continue","description":"Resume execution of a paused system. The system runs until a breakpoint/watchpoint is hit or pause is called. Poll get_status to detect when execution stops.","inputSchema":{"type":"object","properties":{}},"title":"Continue","annotations":{"readOnlyHint":false,"destructiveHint":false,"idempotentHint":false,"openWorldHint":false}},
{"name":"read_memory","description":"Read bytes from memory at a given address. Returns: {address, size, data} where data is a hex or base64 encoded string.","inputSchema":{"type":"object","properties":{"address":{"description":"Start address (integer or hex string)"},"size":{"type":"integer","description":"Number of bytes to read (max 1048576)"},"format":{"type":"string","enum":["hex","base64"],"default":"hex","description":"Output format for the data"}},"required":["address","size"]},"title":"Read Memory","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"write_memory","description":"Write bytes to memory at a given address. Returns: {address, bytes_written}.","inputSchema":{"type":"object","properties":{"address":{"description":"Start address (integer or hex string)"},"data":{"type":"string","description":"Data to write (hex or base64 encoded string)"},"format":{"type":"string","enum":["hex","base64"],"default":"hex","description":"Encoding format of the data parameter"}},"required":["address","data"]},"title":"Write Memory","annotations":{"readOnlyHint":false,"destructiveHint":true,"idempotentHint":false,"openWorldHint":false}},
{"name":"search_memory","description":"Search PS1 memory for a byte pattern with optional wildcard mask. Returns all matching addresses in one call. For iterative value hunting (cheat search) use memory_scan_start instead. Returns: {matches: [\"0x80012345\", ...]} array of hex address strings (max 100).","inputSchema":{"type":"object","properties":{"start":{"description":"Start address (integer or hex string, default 0)"},"pattern":{"type":"string","description":"Hex string pattern to search for (e.g. 'FF0042')"},"mask":{"type":"string","description":"Hex string mask (FF=must match, 00=wildcard). Same length as pattern. Default: all FF"}},"required":["pattern"]},"title":"Search Memory","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"dump_ram","description":"Dump the entire RAM (2MB) to a file on disk. Returns: {path, size} confirming the output file and byte count.","inputSchema":{"type":"object","properties":{"path":{"type":"string","description":"File path to save the RAM dump to (required)"}},"required":["path"]},"title":"Dump RAM","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"get_gpu_state","description":"Get GPU state. aspect='registers' (default): GPUSTAT, display resolution, interlace mode. aspect='draw': texture page, semi-transparency, dithering, draw area. aspect='stats': rendered resolution, refresh frequencies, aspect ratio. aspect='crtc': beam position, horizontal/vertical timing, display rects. aspect='all': everything. Returns: JSON object with fields specific to the requested aspect.","inputSchema":{"type":"object","properties":{"aspect":{"type":"string","enum":["registers","draw","stats","crtc","all"],"default":"registers","description":"Which aspect of GPU state to return"}}},"title":"Get GPU State","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"get_spu_state","description":"Get SPU state. detail='basic' (default): control/status registers, main volume, CD volume. detail='voices': per-voice state (volume, pitch, ADSR, addresses) for all 24 voices (or specific voice). detail='reverb': reverb configuration (all 32 reverb registers, work area, volumes). Returns: JSON object with fields specific to the requested detail level.","inputSchema":{"type":"object","properties":{"detail":{"type":"string","enum":["basic","voices","reverb"],"default":"basic","description":"Detail level"},"voice":{"type":"integer","minimum":0,"maximum":23,"description":"Optional voice index (0-23) for voices detail"}}},"title":"Get SPU State","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"get_cdrom_state","description":"Get CD-ROM state. detail='basic' (default): disc inserted, region, current track, seek position, drive state. detail='extended': adds precise LBA position, MSF time, full track layout, sub-Q channel data, total disc size. Returns: JSON object with has_media, disc_region, current_lba, track_count, and more depending on detail level.","inputSchema":{"type":"object","properties":{"detail":{"type":"string","enum":["basic","extended"],"default":"basic","description":"Detail level"}}},"title":"Get CDROM State","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"dump_vram","description":"Dump the entire 1024x512 16-bit VRAM to a file on disk (PNG or raw binary). Returns: {path, format, size} confirming the output file.","inputSchema":{"type":"object","properties":{"path":{"type":"string","description":"File path to save the VRAM dump to (required)"},"format":{"type":"string","enum":["png","bin"],"default":"png","description":"Output format: 'png' (RGBA8 PNG image) or 'bin' (raw 16-bit VRAM data)"}},"required":["path"]},"title":"Dump VRAM","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"dump_spu_ram","description":"Dump the entire 512KB SPU RAM to a file on disk. Returns: {path, size} confirming the output file.","inputSchema":{"type":"object","properties":{"path":{"type":"string","description":"File path to save the SPU RAM dump to (required)"}},"required":["path"]},"title":"Dump SPU RAM","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"get_status","description":"Get system status. detail='basic' (default): state, title, serial, frame number, tick counter (works when no game loaded). detail='full': adds game_path, region, speed, FPS, taints, latency, session time, boot mode. Returns: JSON object with state ('running'/'paused'/'shutdown'), game_serial, game_title, frame_number, and more depending on detail level.","inputSchema":{"type":"object","properties":{"detail":{"type":"string","enum":["basic","full"],"default":"basic","description":"Detail level"}}},"title":"Get Status","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"reset","description":"Reset the running system","inputSchema":{"type":"object","properties":{}},"title":"Reset","annotations":{"readOnlyHint":false,"destructiveHint":true,"idempotentHint":false,"openWorldHint":false}},
{"name":"save_state","description":"Save system state to a numbered slot. Returns: {status: 'saved', slot}.","inputSchema":{"type":"object","properties":{"slot":{"type":"integer","minimum":1,"maximum":10,"description":"Save state slot number (1-10)"}},"required":["slot"]},"title":"Save State","annotations":{"readOnlyHint":false,"destructiveHint":false,"idempotentHint":false,"openWorldHint":false}},
{"name":"load_state","description":"Load system state from a numbered slot. Returns: {status: 'loaded', slot}.","inputSchema":{"type":"object","properties":{"slot":{"type":"integer","minimum":1,"maximum":10,"description":"Save state slot number (1-10)"}},"required":["slot"]},"title":"Load State","annotations":{"readOnlyHint":false,"destructiveHint":false,"idempotentHint":false,"openWorldHint":false}},
{"name":"frame_step","description":"Advance emulation by exactly one frame, then pause. Call repeatedly for multiple frames. Returns: {status: 'stepped', frame_number}.","inputSchema":{"type":"object","properties":{}},"title":"Frame Step","annotations":{"readOnlyHint":false,"destructiveHint":false,"idempotentHint":false,"openWorldHint":false}},
{"name":"press_button","description":"Press a controller button (optionally auto-release after N frames). Returns: {slot, button, state: 'pressed', auto_release_frames?}.","inputSchema":{"type":"object","properties":{"slot":{"type":"integer","default":0,"description":"Controller slot (0-7)"},"button":{"type":"string","enum":["Cross","Circle","Square","Triangle","Up","Down","Left","Right","L1","L2","R1","R2","Start","Select","L3","R3"],"description":"Button name"},"duration_frames":{"type":"integer","description":"Auto-release after this many frames (optional)"}},"required":["button"]},"title":"Press Button","annotations":{"readOnlyHint":false,"destructiveHint":false,"idempotentHint":false,"openWorldHint":false}},
{"name":"release_button","description":"Release a previously pressed controller button. Returns: {slot, button, state: 'released'}.","inputSchema":{"type":"object","properties":{"slot":{"type":"integer","default":0,"description":"Controller slot (0-7)"},"button":{"type":"string","enum":["Cross","Circle","Square","Triangle","Up","Down","Left","Right","L1","L2","R1","R2","Start","Select","L3","R3"],"description":"Button name to release"}},"required":["button"]},"title":"Release Button","annotations":{"readOnlyHint":false,"destructiveHint":false,"idempotentHint":false,"openWorldHint":false}},
{"name":"set_analog","description":"Set analog stick position (for DualShock controllers). Returns: {slot, stick, x, y}.","inputSchema":{"type":"object","properties":{"slot":{"type":"integer","default":0,"description":"Controller slot (0-7)"},"stick":{"type":"string","enum":["left","right"],"description":"Which analog stick"},"x":{"type":"number","description":"X axis value (-1.0 to 1.0)"},"y":{"type":"number","description":"Y axis value (-1.0 to 1.0)"}},"required":["stick","x","y"]},"title":"Set Analog","annotations":{"readOnlyHint":false,"destructiveHint":false,"idempotentHint":false,"openWorldHint":false}},
{"name":"get_controller_state","description":"Get current controller button and analog state. Returns: {slot, type, buttons: {name: bool, ...}, analog?: {byte0..byte3}} where buttons are true if pressed.","inputSchema":{"type":"object","properties":{"slot":{"type":"integer","default":0,"description":"Controller slot (0-7)"}}},"title":"Get Controller State","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"list_controllers","description":"List all configured controllers with their types and available bindings. Returns: {controllers: [{slot, type, display_name, bindings: [{name, display_name, type, bind_index}]}]}.","inputSchema":{"type":"object","properties":{}},"title":"List Controllers","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"input_sequence","description":"Execute a timed sequence of button inputs across multiple frames. Returns: {sequence_id, total_frames}.","inputSchema":{"type":"object","properties":{"slot":{"type":"integer","default":0,"description":"Controller slot (0-7)"},"sequence":{"type":"array","description":"Array of input steps","items":{"type":"object","properties":{"buttons":{"type":"array","items":{"type":"string"},"description":"Buttons to press for this step"},"duration_frames":{"type":"integer","description":"How many frames to hold these buttons"}}}}},"required":["sequence"]},"title":"Input Sequence","annotations":{"readOnlyHint":false,"destructiveHint":false,"idempotentHint":false,"openWorldHint":false}},
{"name":"get_settings","description":"Get current emulator settings (GPU renderer, resolution scale, CPU overclock, etc.). Returns: JSON object grouped by section {GPU: {...}, Display: {...}, CPU: {...}, Emulation: {...}, Audio: {...}, CDROM: {...}}.","inputSchema":{"type":"object","properties":{}},"title":"Get Settings","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"set_setting","description":"Set an emulator setting. Supported: GPU/ResolutionScale (1-16), GPU/PGXP (bool), CPU/Overclock (bool), CPU/OverclockNumerator (uint), CPU/OverclockDenominator (uint), Audio/OutputVolume (0-100), Audio/Muted (bool), CDROM/ReadSpeedup (1-99), CDROM/SeekSpeedup (1-99). Returns: {setting, applied: bool}.","inputSchema":{"type":"object","properties":{"section":{"type":"string","enum":["GPU","CPU","Audio","CDROM"],"description":"Settings section"},"key":{"type":"string","description":"Setting key name"},"value":{"description":"Value to set (string, number, or boolean)"}},"required":["section","key","value"]},"title":"Set Setting","annotations":{"readOnlyHint":false,"destructiveHint":false,"idempotentHint":false,"openWorldHint":false}},
{"name":"set_speed","description":"Control emulation speed, fast-forward, turbo, or rewind. Returns: {target_speed, fast_forward, turbo, rewind} reflecting the current speed state after changes.","inputSchema":{"type":"object","properties":{"speed":{"type":"number","description":"Speed multiplier (e.g. 1.0 = normal, 2.0 = double)"},"fast_forward":{"type":"boolean","description":"Enable/disable fast-forward"},"turbo":{"type":"boolean","description":"Enable/disable turbo mode"},"rewind":{"type":"boolean","description":"Enable/disable rewind"}}},"title":"Set Speed","annotations":{"readOnlyHint":false,"destructiveHint":false,"idempotentHint":false,"openWorldHint":false}},
{"name":"take_screenshot","description":"Take a screenshot and save it to disk. Returns: {status: 'screenshot_saved', path?}.","inputSchema":{"type":"object","properties":{"path":{"type":"string","description":"File path to save screenshot (optional, uses default if not specified)"}}},"title":"Take Screenshot","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"list_cheats","description":"List available cheats and patches for the current game. Returns: {codes: [{name, author, description, type, activation, from_database}], count}.","inputSchema":{"type":"object","properties":{"type":{"type":"string","enum":["cheats","patches","all"],"default":"all","description":"Filter by type: cheats, patches, or all"}}},"title":"List Cheats","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"toggle_cheat","description":"Toggle a cheat code on or off by name. Returns: {name, enabled, cheats_master_enabled}.","inputSchema":{"type":"object","properties":{"name":{"type":"string","description":"Name of the cheat to toggle"},"enabled":{"type":"boolean","description":"Force enabled (true) or disabled (false). Omit to toggle."}},"required":["name"]},"title":"Toggle Cheat","annotations":{"readOnlyHint":false,"destructiveHint":false,"idempotentHint":false,"openWorldHint":false}},
{"name":"apply_cheat","description":"Apply a cheat code to memory once (one-shot write). Unlike toggle_cheat which enables persistent per-frame codes, this writes the cheat values to RAM immediately and does not persist. Returns: {status: 'applied', name}.","inputSchema":{"type":"object","properties":{"name":{"type":"string","description":"Name of the cheat to apply"}},"required":["name"]},"title":"Apply Cheat","annotations":{"readOnlyHint":false,"destructiveHint":false,"idempotentHint":false,"openWorldHint":false}},
{"name":"get_cheat_status","description":"Get the status of all active cheats. Returns: {cheats_enabled, active_cheat_count, active_patch_count, widescreen_patch_active}.","inputSchema":{"type":"object","properties":{}},"title":"Get Cheat Status","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"list_save_states","description":"List available save states for the current or specified game. Returns: {serial, count, states: [{path, slot, global, timestamp}]}.","inputSchema":{"type":"object","properties":{"serial":{"type":"string","description":"Game serial to list save states for (optional, uses current game)"}}},"title":"List Save States","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"swap_memory_cards","description":"Swap memory cards between slot 1 and slot 2","inputSchema":{"type":"object","properties":{}},"title":"Swap Memory Cards","annotations":{"readOnlyHint":false,"destructiveHint":false,"idempotentHint":false,"openWorldHint":false}},
{"name":"boot_game","description":"Boot a game from a disc image or executable path. Returns: {status: 'booted', path}.","inputSchema":{"type":"object","properties":{"path":{"type":"string","description":"Path to the game disc image or executable"},"fast_boot":{"type":"boolean","description":"Skip BIOS intro (fast boot)"},"start_paused":{"type":"boolean","description":"Start the system in paused state"},"force_software":{"type":"boolean","description":"Force software renderer"}},"required":["path"]},"title":"Boot Game","annotations":{"readOnlyHint":false,"destructiveHint":false,"idempotentHint":false,"openWorldHint":false}},
{"name":"shutdown_system","description":"Shut down the running system. Returns: {status: 'shutdown', save_resume_state}.","inputSchema":{"type":"object","properties":{"save_resume_state":{"type":"boolean","default":true,"description":"Save a resume state before shutting down"}}},"title":"Shutdown System","annotations":{"readOnlyHint":false,"destructiveHint":true,"idempotentHint":false,"openWorldHint":false}},
{"name":"get_cop0_state","description":"Get COP0 (System Control Coprocessor) registers with decoded bitfields: Status Register, CAUSE, EPC, DCIC breakpoint control, etc. Essential for interrupt and exception analysis. Returns: {sr: {bits, IEc, KUc, ...}, cause: {bits, Excode, Excode_name, ...}, dcic: {bits, ...}, EPC, BadVaddr, BPC, BDA, BPCM, BDAM, TAR, PRID}.","inputSchema":{"type":"object","properties":{}},"title":"Get COP0 State","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"get_cpu_execution_state","description":"Get CPU pipeline/execution state (PC, NPC, branch delay slot, load delays, ticks, cache control). Returns: {pc, npc, current_instruction_pc, in_branch_delay_slot, pending_ticks, downcount, load_delay_reg/value, cache_control, ...}.","inputSchema":{"type":"object","properties":{}},"title":"Get CPU Execution State","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"get_gte_registers","description":"Read all 64 GTE (Geometry Transform Engine) registers: 32 data registers (vectors, matrices, color FIFO) + 32 control registers (rotation matrix, translation, projection). Used for 3D math on PS1. Returns: {data_registers: {name: hex, ...}, control_registers: {name: hex, ...}}.","inputSchema":{"type":"object","properties":{}},"title":"Get GTE Registers","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"set_gte_register","description":"Write a value to a GTE register by index (0-63). Returns: {success, index, value}.","inputSchema":{"type":"object","properties":{"index":{"type":"integer","minimum":0,"maximum":63,"description":"GTE register index (0-31 data, 32-63 control)"},"value":{"description":"Value to write (integer or hex string)"}},"required":["index","value"]},"title":"Set GTE Register","annotations":{"readOnlyHint":false,"destructiveHint":true,"idempotentHint":false,"openWorldHint":false}},
{"name":"get_memory_map","description":"Get PS1 memory region layout (RAM, BIOS, EXP1, scratchpad, hardware registers). Returns: {ram_size, memory_regions: [{name, start, end, size, writable}], hardware_registers: [{name, start, size}]}.","inputSchema":{"type":"object","properties":{}},"title":"Get Memory Map","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"inject_executable","description":"Load a PS1 executable (PS-EXE) into emulated RAM at the address specified in its header. Optionally sets PC to the entry point to begin execution. Returns: {success, path, size, set_pc}.","inputSchema":{"type":"object","properties":{"path":{"type":"string","description":"Path to the PS-EXE file"},"set_pc":{"type":"boolean","default":true,"description":"Set PC to the executable entry point"}},"required":["path"]},"title":"Inject Executable","annotations":{"readOnlyHint":false,"destructiveHint":true,"idempotentHint":false,"openWorldHint":false}},
{"name":"get_achievements_state","description":"Get RetroAchievements state. detail='basic' (default): login status, current game ID, hardcore mode, rich presence. detail='full': adds game metadata, achievement list, detailed login info. Returns: {active, logged_in, hardcore_mode, has_active_game, username?, game_id?, game_title?, rich_presence?, ...}.","inputSchema":{"type":"object","properties":{"detail":{"type":"string","enum":["basic","full"],"default":"basic","description":"Detail level"}}},"title":"Get Achievements State","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"get_cpu_icache_state","description":"Get CPU instruction cache internals: cache control register, line tags, validity bits, cached words. Useful for diagnosing cache-related bugs in PS1 software. Returns: {cache_control, icache_enabled, total_lines, valid_lines, cache_lines: [{line, tag, tag_address, valid, words: [...]}]}.","inputSchema":{"type":"object","properties":{"address":{"description":"Optional address to filter cache lines (shows only matching line)"}}},"title":"Get CPU ICache State","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"get_pgxp_state","description":"Get PGXP state and settings. PGXP tracks sub-pixel vertex precision through the CPU pipeline to reduce PS1 polygon jitter. Returns: {pgxp_enabled, pgxp_culling, pgxp_texture_correction, pgxp_vertex_cache, pgxp_cpu_mode, pgxp_depth_buffer, pgxp_tolerance, ...}.","inputSchema":{"type":"object","properties":{}},"title":"Get PGXP State","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"list_games","description":"List all games in the game library with optional filtering and sorting. Returns: {total_entries, matched_entries, returned_entries, games: [{title, serial, path, region, size, ...}]}.","inputSchema":{"type":"object","properties":{"filter":{"type":"string","description":"Filter by substring match on title, serial, or path"},"sort_by":{"type":"string","enum":["title","serial","region","size","last_played"],"default":"title","description":"Sort field"},"max_results":{"type":"integer","default":100,"description":"Maximum number of results to return"}}},"title":"List Games","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"get_game_info","description":"Get detailed information about a game by serial or path. If neither provided, returns info for the currently loaded game including database entry, cover path, and play time. Returns: {game: {title, serial, path, region, size, ...}, cover_image_path}.","inputSchema":{"type":"object","properties":{"serial":{"type":"string","description":"Game serial (e.g. 'SLUS-01042')"},"path":{"type":"string","description":"Path to the game disc image"}}},"title":"Get Game Info","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"refresh_game_list","description":"Rescan game directories and refresh the game list. Returns: {status: 'refreshed', invalidate_cache, total_entries}.","inputSchema":{"type":"object","properties":{"invalidate_cache":{"type":"boolean","default":false,"description":"Invalidate the cache and rescan all files"}}},"title":"Refresh Game List","annotations":{"readOnlyHint":false,"destructiveHint":false,"idempotentHint":false,"openWorldHint":false}},
{"name":"list_bios","description":"List all available BIOS images with region and info. Returns: {bios_directory, count, images: [{filename, description, region, hash, supports_fast_boot, priority}]}.","inputSchema":{"type":"object","properties":{}},"title":"List BIOS","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"get_bios_info","description":"Get detailed information about a specific BIOS file. Returns: {filename, bios_directory, description, region, hash, supports_fast_boot, region_check, priority, fastboot_patch_type}.","inputSchema":{"type":"object","properties":{"filename":{"type":"string","description":"BIOS filename"}},"required":["filename"]},"title":"Get BIOS Info","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"get_save_state_info","description":"Get extended information about a save state (title, serial, timestamp, screenshot info). Returns: {slot, global, path, title, serial, timestamp, has_screenshot, screenshot_width, screenshot_height, ...}.","inputSchema":{"type":"object","properties":{"slot":{"type":"integer","minimum":1,"maximum":10,"description":"Save state slot number"},"global":{"type":"boolean","default":false,"description":"Use global save state instead of per-game"},"serial":{"type":"string","description":"Game serial (uses current game if omitted)"}},"required":["slot"]},"title":"Get Save State Info","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"delete_save_states","description":"Delete all save states for a game by serial. Returns: {status: 'deleted', serial, deleted_count}.","inputSchema":{"type":"object","properties":{"serial":{"type":"string","description":"Game serial"},"include_resume":{"type":"boolean","default":false,"description":"Also delete resume save state"}},"required":["serial"]},"title":"Delete Save States","annotations":{"readOnlyHint":false,"destructiveHint":true,"idempotentHint":false,"openWorldHint":false}},
{"name":"undo_load_state","description":"Undo the last state load, restoring the state before loading. Returns: {status: 'undone'}.","inputSchema":{"type":"object","properties":{}},"title":"Undo Load State","annotations":{"readOnlyHint":false,"destructiveHint":false,"idempotentHint":false,"openWorldHint":false}},
{"name":"get_save_state_screenshot","description":"Extract the screenshot from a save state and save it as a temporary PNG file. Returns: {slot, global, screenshot_path, width, height}.","inputSchema":{"type":"object","properties":{"slot":{"type":"integer","minimum":1,"maximum":10,"description":"Save state slot number"},"global":{"type":"boolean","default":false,"description":"Use global save state"},"serial":{"type":"string","description":"Game serial (uses current game if omitted)"}},"required":["slot"]},"title":"Get Save State Screenshot","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"get_cheat_details","description":"Get full details of a cheat code by name (body, author, options). Returns: {name, author, description, body, type, activation, from_database, options?: [{name, value}]}.","inputSchema":{"type":"object","properties":{"name":{"type":"string","description":"Cheat code name"}},"required":["name"]},"title":"Get Cheat Details","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"create_cheat","description":"Create a new cheat code for the current game. Returns: {status: 'created', name, path, type, activation}.","inputSchema":{"type":"object","properties":{"name":{"type":"string","description":"Cheat name"},"body":{"type":"string","description":"Cheat code body (GameShark format)"},"type":{"type":"string","default":"gameshark","description":"Code type"},"activation":{"type":"string","enum":["manual","end_frame"],"default":"manual","description":"When the code is applied"}},"required":["name","body"]},"title":"Create Cheat","annotations":{"readOnlyHint":false,"destructiveHint":false,"idempotentHint":false,"openWorldHint":false}},
{"name":"import_cheats","description":"Import cheat codes from text in various formats. Returns: {status: 'imported', count, codes: [{name, type, activation}]}.","inputSchema":{"type":"object","properties":{"content":{"type":"string","description":"Cheat code text content"},"format":{"type":"string","enum":["duckstation","pcsx","libretro","epsxe"],"description":"Input format"}},"required":["content","format"]},"title":"Import Cheats","annotations":{"readOnlyHint":false,"destructiveHint":false,"idempotentHint":false,"openWorldHint":false}},
{"name":"export_cheats","description":"Export all cheats for the current game to a file. Returns: {status: 'exported', path, count}.","inputSchema":{"type":"object","properties":{"path":{"type":"string","description":"Output file path"}},"required":["path"]},"title":"Export Cheats","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"validate_cheat","description":"Check if a cheat code body is syntactically valid (correct format, valid addresses) without saving it. Returns: {valid: bool, error?: string}.","inputSchema":{"type":"object","properties":{"body":{"type":"string","description":"Cheat code body to validate"},"type":{"type":"string","default":"gameshark","description":"Code type"},"activation":{"type":"string","default":"manual","description":"Activation type"}},"required":["body"]},"title":"Validate Cheat","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"get_memory_card_info","description":"Get memory card information (free blocks, files, save list) for a slot. slot defaults to 0. Returns: {slot, path, valid, free_blocks, total_blocks, file_count, files: [{filename, title, size, num_blocks, deleted}]}.","inputSchema":{"type":"object","properties":{"slot":{"type":"integer","minimum":0,"maximum":1,"default":0,"description":"Memory card slot (0 or 1)"}}},"title":"Get Memory Card Info","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"read_memory_card_file","description":"Read a save file from a PS1 memory card and write it to a temporary file. Returns: {slot, filename, title, size, num_blocks, output_path}.","inputSchema":{"type":"object","properties":{"slot":{"type":"integer","minimum":0,"maximum":1,"description":"Memory card slot"},"filename":{"type":"string","description":"Filename on the memory card"}},"required":["slot","filename"]},"title":"Read Memory Card File","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"delete_memory_card_file","description":"Delete a file from a memory card. Returns: {status: 'deleted', slot, filename, free_blocks}.","inputSchema":{"type":"object","properties":{"slot":{"type":"integer","minimum":0,"maximum":1,"description":"Memory card slot"},"filename":{"type":"string","description":"Filename to delete"}},"required":["slot","filename"]},"title":"Delete Memory Card File","annotations":{"readOnlyHint":false,"destructiveHint":true,"idempotentHint":false,"openWorldHint":false}},
{"name":"export_memory_card_save","description":"Export a save file from a memory card to disk. Returns: {status: 'exported', slot, filename, output_path}.","inputSchema":{"type":"object","properties":{"slot":{"type":"integer","minimum":0,"maximum":1,"description":"Memory card slot"},"filename":{"type":"string","description":"Filename on the memory card"},"output_path":{"type":"string","description":"Output file path"}},"required":["slot","filename","output_path"]},"title":"Export Memory Card Save","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"import_memory_card_save","description":"Import a save file into a memory card. Returns: {status: 'imported', slot, input_path, free_blocks}.","inputSchema":{"type":"object","properties":{"slot":{"type":"integer","minimum":0,"maximum":1,"description":"Memory card slot"},"input_path":{"type":"string","description":"Input save file path"}},"required":["slot","input_path"]},"title":"Import Memory Card Save","annotations":{"readOnlyHint":false,"destructiveHint":false,"idempotentHint":false,"openWorldHint":false}},
{"name":"list_shaders","description":"List all available post-processing shaders. Returns: {count, shaders: [{name, type}]}.","inputSchema":{"type":"object","properties":{}},"title":"List Shaders","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"get_shader_chain","description":"Get the current post-processing shader chain configuration. Returns: {enabled, stage_count, stages: [{index, shader_name, enabled}]}.","inputSchema":{"type":"object","properties":{}},"title":"Get Shader Chain","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"add_shader","description":"Add a shader to the post-processing chain. Returns: {status: 'added', shader_name, index}.","inputSchema":{"type":"object","properties":{"shader_name":{"type":"string","description":"Name of the shader to add"}},"required":["shader_name"]},"title":"Add Shader","annotations":{"readOnlyHint":false,"destructiveHint":false,"idempotentHint":false,"openWorldHint":false}},
{"name":"remove_shader","description":"Remove a shader from the post-processing chain by index. Returns: {status: 'removed', index}.","inputSchema":{"type":"object","properties":{"index":{"type":"integer","description":"Stage index to remove"}},"required":["index"]},"title":"Remove Shader","annotations":{"readOnlyHint":false,"destructiveHint":false,"idempotentHint":false,"openWorldHint":false}},
{"name":"get_shader_options","description":"Get configurable options for a shader stage. Returns: {index, shader_name, option_count, options: [{name, ui_name, type, value, default_value, min_value?, max_value?, step_value?, choices?}]}.","inputSchema":{"type":"object","properties":{"index":{"type":"integer","description":"Stage index"}},"required":["index"]},"title":"Get Shader Options","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"set_shader_option","description":"Set a shader option value. Returns: {status: 'set', index, option_name, value}.","inputSchema":{"type":"object","properties":{"index":{"type":"integer","description":"Stage index"},"option_name":{"type":"string","description":"Option name"},"value":{"description":"New value (number, boolean, or array)"}},"required":["index","option_name","value"]},"title":"Set Shader Option","annotations":{"readOnlyHint":false,"destructiveHint":false,"idempotentHint":false,"openWorldHint":false}},
{"name":"list_hotkeys","description":"List all available hotkeys. Returns: {hotkeys: [{name, category, display_name}], count}.","inputSchema":{"type":"object","properties":{"category":{"type":"string","description":"Filter by category name"}}},"title":"List Hotkeys","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"trigger_hotkey","description":"Trigger a hotkey by name (simulates press and release). Returns: {status: 'triggered', name, display_name}.","inputSchema":{"type":"object","properties":{"name":{"type":"string","description":"Hotkey name"}},"required":["name"]},"title":"Trigger Hotkey","annotations":{"readOnlyHint":false,"destructiveHint":false,"idempotentHint":false,"openWorldHint":false}},
{"name":"get_disc_info","description":"Get disc track information for the current media. Returns: {media_path, disc_region, is_ps1_disc, is_audio_cd, track_count, tracks: [{track_number, mode, start_lba, length, bytes_per_sector, is_data}]}.","inputSchema":{"type":"object","properties":{}},"title":"Get Disc Info","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"list_disc_files","description":"List files and directories on the game disc ISO9660 filesystem. Returns: {path, entries: [{name, is_directory, size, lba, sectors}], count}.","inputSchema":{"type":"object","properties":{"path":{"type":"string","default":"/","description":"Directory path to list (default: root)"},"recursive":{"type":"boolean","default":false,"description":"Recursively list subdirectories"}}},"title":"List Disc Files","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"read_disc_file","description":"Read a file from the game disc ISO9660 filesystem and save it to a temporary file. Max file size: 16MB. Returns: {path, size, output_path}.","inputSchema":{"type":"object","properties":{"path":{"type":"string","description":"File path on the disc (e.g. '/PSX.EXE', '/DATA/INGS.BIN')"}},"required":["path"]},"title":"Read Disc File","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"read_disc_sectors","description":"Read raw CD-ROM sectors by LBA (2352 bytes each) and save to a temporary file. Max 128 sectors per call. Returns: {lba, count, size, output_path}.","inputSchema":{"type":"object","properties":{"lba":{"type":"integer","description":"Starting logical block address"},"count":{"type":"integer","default":1,"description":"Number of sectors to read (max 128)"}},"required":["lba"]},"title":"Read Disc Sectors","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"read_vram_region","description":"Read a rectangular region of PS1 VRAM (1024x512, 16-bit) and save to a temporary file. Format: png (visual) or raw (u16 pixels for analysis). Returns: {x, y, width, height, format, output_path}.","inputSchema":{"type":"object","properties":{"x":{"type":"integer","description":"Left edge (0-1023)"},"y":{"type":"integer","description":"Top edge (0-511)"},"width":{"type":"integer","description":"Width in pixels"},"height":{"type":"integer","description":"Height in pixels"},"format":{"type":"string","enum":["png","raw"],"default":"png","description":"Output format: png (RGBA8) or raw (u16 1555 ABGR)"}},"required":["x","y","width","height"]},"title":"Read VRAM Region","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"write_vram_region","description":"Write raw u16 pixel data from a file into a VRAM region. Returns: {x, y, width, height, status: 'written'}.","inputSchema":{"type":"object","properties":{"x":{"type":"integer","description":"Left edge (0-1023)"},"y":{"type":"integer","description":"Top edge (0-511)"},"width":{"type":"integer","description":"Width in pixels"},"height":{"type":"integer","description":"Height in pixels"},"input_path":{"type":"string","description":"Path to raw u16 pixel data file (must be width*height*2 bytes)"},"format":{"type":"string","enum":["raw"],"default":"raw","description":"Input format (currently only raw supported)"}},"required":["x","y","width","height","input_path"]},"title":"Write VRAM Region","annotations":{"readOnlyHint":false,"destructiveHint":true,"idempotentHint":false,"openWorldHint":false}},
{"name":"snapshot_memory","description":"Take a snapshot of PS1 RAM for later comparison. Workflow: 1) snapshot_memory, 2) let the game run or perform actions, 3) diff_memory to see what changed. Returns: {address, physical_address, size}.","inputSchema":{"type":"object","properties":{"address":{"type":"string","default":"0x80000000","description":"Start address (hex string, default: start of RAM)"},"size":{"type":"integer","description":"Number of bytes to snapshot (default: entire RAM from address)"}}},"title":"Snapshot Memory","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"diff_memory","description":"Compare current PS1 RAM with the last snapshot from snapshot_memory. Must call snapshot_memory first. Returns: {address, size, changes: [{offset, address, old_value, new_value}], total_changes, truncated} (max 500 changes shown).","inputSchema":{"type":"object","properties":{}},"title":"Diff Memory","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"find_free_ram","description":"Scan PS1 RAM for contiguous zero-byte regions, sorted by size descending. Useful for finding free space to inject custom code or data. Returns: {min_size, count, regions: [{address, size}]} (max 50 regions).","inputSchema":{"type":"object","properties":{"min_size":{"type":"integer","default":256,"description":"Minimum contiguous zero bytes to report"},"start":{"type":"string","default":"0x80010000","description":"Scan start address (hex string)"},"end":{"type":"string","description":"Scan end address (hex string, default: end of RAM)"}}},"title":"Find Free RAM","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"breakpoint","description":"Manage breakpoints and watchpoints. Actions: add (create), remove (delete), list (show all), enable/disable (toggle without removing), clear (remove all). For add/remove/enable/disable: type and address are required. Returns: For add: {address, type}. For list: array of {address, type, enabled, hit_count}. For enable/disable: {success, address, type, enabled}. For clear: {success}.","inputSchema":{"type":"object","properties":{"action":{"type":"string","enum":["add","remove","list","enable","disable","clear"],"description":"Action to perform"},"type":{"type":"string","enum":["execute","read","write"],"default":"execute","description":"Breakpoint type (for add/remove/enable/disable)"},"address":{"description":"Address (integer or hex string, for add/remove/enable/disable)"}},"required":["action"]},"title":"Breakpoint","annotations":{"readOnlyHint":false,"destructiveHint":false,"idempotentHint":false,"openWorldHint":false}},
{"name":"get_hardware_state","description":"Get hardware subsystem state. subsystem='dma': DMA controller (all 7 channels, DPCR, DICR). subsystem='timers': hardware timers (counter, target, clock source). subsystem='interrupts': interrupt controller (per-IRQ status, mask, active). subsystem='mdec': Motion Decoder (active, decoding, FIFO, current block). subsystem='timing_events': all active timing events (period, interval, next tick). Returns: JSON object with subsystem-specific fields.","inputSchema":{"type":"object","properties":{"subsystem":{"type":"string","enum":["dma","timers","interrupts","mdec","timing_events"],"description":"Hardware subsystem to query"}},"required":["subsystem"]},"title":"Get Hardware State","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
{"name":"memory_scan","description":"Iterative memory scanner (cheat search). Workflow: start to find initial matches, refine to narrow down, results to read matches, reset to clear. For one-shot pattern search use search_memory instead. Returns: For start/refine: {status, result_count}. For results: {total_results, offset, count, results: [{address, value, last_value, first_value, value_changed}]}. For reset: {status, result_count: 0}.","inputSchema":{"type":"object","properties":{"action":{"type":"string","enum":["start","refine","results","reset"],"description":"Action to perform"},"value":{"description":"Value to search for (for start/refine)"},"size":{"type":"string","enum":["byte","halfword","word"],"default":"word","description":"Memory access size (for start)"},"operator":{"type":"string","enum":["equal","not_equal","less_than","less_equal","greater_than","greater_equal","any","changed","decreased","increased","less_than_last","less_equal_last","greater_than_last","greater_equal_last","not_equal_last","equal_last"],"description":"Comparison operator (for start/refine)"},"signed":{"type":"boolean","default":false,"description":"Treat values as signed (for start)"},"start":{"type":"string","description":"Start address hex (for start)"},"end":{"type":"string","description":"End address hex (for start)"},"offset":{"type":"integer","default":0,"description":"Result offset (for results)"},"count":{"type":"integer","default":100,"description":"Number of results (for results)"}},"required":["action"]},"title":"Memory Scan","annotations":{"readOnlyHint":false,"destructiveHint":false,"idempotentHint":false,"openWorldHint":false}},
{"name":"memory_watch","description":"Manage memory address watches. Actions: add (create watch), remove (delete watch by address), list (show all watches with current values). Returns: For add: {status, address, size, freeze, total_watches}. For remove: {status, address, total_watches}. For list: {count, watches: [{description, address, value, size, is_signed, freeze, changed}]}.","inputSchema":{"type":"object","properties":{"action":{"type":"string","enum":["add","remove","list"],"description":"Action to perform"},"address":{"description":"Memory address (for add/remove)"},"size":{"type":"string","enum":["byte","halfword","word"],"default":"word","description":"Access size (for add)"},"description":{"type":"string","description":"Watch description (for add)"},"signed":{"type":"boolean","default":false,"description":"Treat as signed (for add)"},"freeze":{"type":"boolean","default":false,"description":"Freeze value (for add)"}},"required":["action"]},"title":"Memory Watch","annotations":{"readOnlyHint":false,"destructiveHint":false,"idempotentHint":false,"openWorldHint":false}},
{"name":"vram_watch","description":"Manage VRAM write watchpoints. Actions: add (set watchpoint on VRAM region), remove (delete by ID), list (show all), last_hit (get PC and coords of last hit). Returns: For add: {id, x, y, width, height}. For remove: {status, id}. For list: {watches: [{id, x, y, width, height}]}. For last_hit: {pc, x, y, width, height, regs: {...}, stack: [...]}.","inputSchema":{"type":"object","properties":{"action":{"type":"string","enum":["add","remove","list","last_hit"],"description":"Action to perform"},"x":{"type":"integer","minimum":0,"maximum":1023,"description":"Left edge (for add)"},"y":{"type":"integer","minimum":0,"maximum":511,"description":"Top edge (for add)"},"width":{"type":"integer","minimum":1,"maximum":1024,"description":"Width (for add)"},"height":{"type":"integer","minimum":1,"maximum":512,"description":"Height (for add)"},"id":{"type":"integer","description":"Watchpoint ID (for remove)"}},"required":["action"]},"title":"VRAM Watch","annotations":{"readOnlyHint":false,"destructiveHint":false,"idempotentHint":false,"openWorldHint":false}},
{"name":"capture","description":"Control video/audio capture recording. Actions: start (begin recording), stop (end recording), status (check if capturing). Returns: For start: {status, path, capturing_audio, capturing_video}. For stop: {status}. For status: {active, path?, video_width?, video_height?, video_fps?, elapsed_time_seconds?}.","inputSchema":{"type":"object","properties":{"action":{"type":"string","enum":["start","stop","status"],"description":"Action to perform"},"path":{"type":"string","description":"Output file path (for start, optional)"}},"required":["action"]},"title":"Capture","annotations":{"readOnlyHint":false,"destructiveHint":false,"idempotentHint":false,"openWorldHint":false}},
{"name":"trace","description":"Control CPU instruction trace logging. Actions: start (begin logging), stop (end logging), status (check if tracing). Warning: generates massive output. Returns: {success?, tracing: bool}.","inputSchema":{"type":"object","properties":{"action":{"type":"string","enum":["start","stop","status"],"description":"Action to perform"}},"required":["action"]},"title":"Trace","annotations":{"readOnlyHint":false,"destructiveHint":false,"idempotentHint":false,"openWorldHint":false}},
{"name":"gpu_dump","description":"Control GPU command dump recording. Actions: start (begin recording), stop (end recording). Returns: {success, num_frames?}.","inputSchema":{"type":"object","properties":{"action":{"type":"string","enum":["start","stop"],"description":"Action to perform"},"path":{"type":"string","description":"Output file path (for start, optional)"},"num_frames":{"type":"integer","default":1,"description":"Frames to capture (for start)"}},"required":["action"]},"title":"GPU Dump","annotations":{"readOnlyHint":false,"destructiveHint":false,"idempotentHint":false,"openWorldHint":false}},
{"name":"disc_control","description":"Manage disc operations. Actions: insert (load disc image by path), eject (remove current disc), switch (change disc by index or direction for multi-disc games), list (show all discs in playlist). Returns: For insert: {status, path}. For eject: {status}. For switch: {status, index/direction}. For list: {has_playlist, count?, current_index?, discs?: [{index, title, current}]}.","inputSchema":{"type":"object","properties":{"action":{"type":"string","enum":["insert","eject","switch","list"],"description":"Action to perform"},"path":{"type":"string","description":"Disc image path (for insert)"},"index":{"type":"integer","description":"Disc index 0-based (for switch)"},"direction":{"type":"string","enum":["next","previous"],"description":"Switch direction (for switch)"}},"required":["action"]},"title":"Disc Control","annotations":{"readOnlyHint":false,"destructiveHint":false,"idempotentHint":false,"openWorldHint":false}},
{"name":"wait_for_pause","description":"Check if the system is paused. Returns current state immediately. Use this after step_over, step_out, or continue to check if the system has stopped (hit a breakpoint/watchpoint). Poll this repeatedly until status is 'paused'. Returns: {status: 'paused'|'running', pc?} where pc is included when paused.","inputSchema":{"type":"object","properties":{}},"title":"Wait For Pause","annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}}
]})json";
    // clang-format on
    const std::string response = MakeJsonRpcResponse(id, TOOLS_JSON);
    SendHttpResponse(200, "application/json", response);
  }
  else if (method == "tools/call")
  {
    const JsonValue& params = request["params"];
    if (params.is_null() || !params.is_object())
    {
      const std::string response = MakeJsonRpcError(id, -32600, "Missing 'params' in tools/call");
      SendHttpResponse(200, "application/json", response);
      return;
    }

    const JsonValue& name_val = params["name"];
    if (name_val.is_null() || !name_val.is_string())
    {
      const std::string response = MakeJsonRpcError(id, -32600, "Missing tool 'name' in params");
      SendHttpResponse(200, "application/json", response);
      return;
    }

    const std::string tool_name(name_val.get_string());
    const JsonValue& tool_args = params["arguments"];
    // tool_args may be null if not provided, handlers check contains() which returns false for null

    DEV_LOG("tools/call: tool='{}'", tool_name);

    const ToolResult tool_result = DispatchToolCall(tool_name, tool_args);

    // Notify the frontend (if registered) so it can refresh debugger views.
    if (!tool_result.IsError() && s_state_changed_callback)
      Host::RunOnUIThread(s_state_changed_callback);

    // Wrap in MCP content envelope. Per MCP spec, tool errors are returned as
    // successful JSON-RPC responses with isError: true, NOT as JSON-RPC errors.
    JsonWriter cw;
    cw.StartObject();
    cw.Key("content");
    cw.StartArray();
    cw.StartObject();
    cw.KeyString("type", "text");
    cw.KeyString("text", tool_result.text);
    cw.EndObject();
    cw.EndArray();
    if (tool_result.IsError())
      cw.KeyBool("isError", true);
    cw.EndObject();
    const std::string response = MakeJsonRpcResponse(id, cw.GetOutput());
    SendHttpResponse(200, "application/json", response);
  }
  else if (method == "resources/list")
  {
    // Static resource definitions as raw JSON.
    static constexpr const char RESOURCES_JSON[] = R"json({"resources":[
{"uri":"emulator://status","name":"Emulator Status","description":"Current emulator status including state, game info, and frame count","mimeType":"application/json"},
{"uri":"emulator://registers","name":"CPU Registers","description":"Complete snapshot of all CPU registers (GPR, COP0, GTE)","mimeType":"application/json"},
{"uri":"emulator://memory/{address}/{size}","name":"Memory Read","description":"Read memory at a given address","mimeType":"application/json"},
{"uri":"emulator://memory_map","name":"PS1 Memory Map","description":"Static PS1 memory region layout (RAM, BIOS, EXP1, scratchpad, hardware registers). Never changes.","mimeType":"application/json"}
]})json";
    const std::string response = MakeJsonRpcResponse(id, RESOURCES_JSON);
    SendHttpResponse(200, "application/json", response);
  }
  else if (method == "resources/read")
  {
    const JsonValue& params = request["params"];
    if (params.is_null() || !params.is_object())
    {
      const std::string response = MakeJsonRpcError(id, -32600, "Missing 'params' in resources/read");
      SendHttpResponse(200, "application/json", response);
      return;
    }

    const JsonValue& uri_val = params["uri"];
    if (uri_val.is_null() || !uri_val.is_string())
    {
      const std::string response = MakeJsonRpcError(id, -32600, "Missing 'uri' in params");
      SendHttpResponse(200, "application/json", response);
      return;
    }

    const std::string uri(uri_val.get_string());
    std::string resource_text;
    std::string resource_uri = uri;

    if (uri == "emulator://status")
    {
      const ToolResult status_result = HandleGetStatus(*JsonValue::Parse("{}"));
      resource_text = status_result.IsError() ? "{}" : status_result.text;
    }
    else if (uri == "emulator://registers")
    {
      const ToolResult reg_result = HandleReadRegisters(*JsonValue::Parse(R"({"group":"all"})"));
      resource_text = reg_result.IsError() ? "{}" : reg_result.text;
    }
    else if (uri.size() > 18 && uri.compare(0, 18, "emulator://memory/") == 0)
    {
      // Parse address and size from URI: emulator://memory/{address}/{size}
      const std::string_view suffix = std::string_view(uri).substr(18); // strlen("emulator://memory/")
      const size_t slash_pos = suffix.find('/');
      if (slash_pos != std::string_view::npos && slash_pos > 0 && slash_pos < suffix.size() - 1)
      {
        const std::string addr_str(suffix.substr(0, slash_pos));
        const std::string size_str(suffix.substr(slash_pos + 1));

        // Parse size as integer.
        const std::optional<u32> size_val = StringUtil::FromChars<u32>(size_str);
        if (size_val.has_value())
        {
          auto mem_args = JsonValue::Parse(
            fmt::format(R"({{"address":"{}","size":{},"format":"hex"}})", addr_str, size_val.value()));
          const ToolResult mem_result = HandleReadMemory(mem_args.value());
          if (mem_result.IsError())
          {
            const std::string response = MakeJsonRpcError(id, -2, mem_result.text);
            SendHttpResponse(200, "application/json", response);
            return;
          }
          resource_text = mem_result.text;
        }
        else
        {
          const std::string response = MakeJsonRpcError(id, -32602, "Invalid size in memory URI");
          SendHttpResponse(200, "application/json", response);
          return;
        }
      }
      else
      {
        const std::string response = MakeJsonRpcError(id, -32602, "Invalid memory URI format. Use: emulator://memory/{address}/{size}");
        SendHttpResponse(200, "application/json", response);
        return;
      }
    }
    else if (uri == "emulator://memory_map")
    {
      const ToolResult map_result = HandleGetMemoryMap(JsonValue());
      if (map_result.IsError())
      {
        const std::string response = MakeJsonRpcError(id, -2, map_result.text);
        SendHttpResponse(200, "application/json", response);
        return;
      }
      resource_text = map_result.text;
    }
    else
    {
      const std::string response = MakeJsonRpcError(id, -32602, fmt::format("Unknown resource URI: {}", uri));
      SendHttpResponse(200, "application/json", response);
      return;
    }

    JsonWriter rw;
    rw.StartObject();
    rw.Key("contents");
    rw.StartArray();
    rw.StartObject();
    rw.KeyString("uri", resource_uri);
    rw.KeyString("mimeType", "application/json");
    rw.KeyString("text", resource_text);
    rw.EndObject();
    rw.EndArray();
    rw.EndObject();
    const std::string response = MakeJsonRpcResponse(id, rw.GetOutput());
    SendHttpResponse(200, "application/json", response);
  }
  else if (method == "resources/subscribe")
  {
    const JsonValue& params = request["params"];
    if (params.is_null() || !params.is_object())
    {
      const std::string response = MakeJsonRpcError(id, -32600, "Missing 'params' in resources/subscribe");
      SendHttpResponse(200, "application/json", response);
      return;
    }

    const JsonValue& uri_val = params["uri"];
    if (uri_val.is_null() || !uri_val.is_string())
    {
      const std::string response = MakeJsonRpcError(id, -32600, "Missing 'uri' in params");
      SendHttpResponse(200, "application/json", response);
      return;
    }

    const std::string uri(uri_val.get_string());
    if (std::find(s_subscribed_resources.begin(), s_subscribed_resources.end(), uri) ==
        s_subscribed_resources.end())
    {
      s_subscribed_resources.push_back(uri);
    }

    const std::string response = MakeJsonRpcResponse(id, "{}");
    SendHttpResponse(200, "application/json", response);
  }
  else if (method == "resources/unsubscribe")
  {
    const JsonValue& params = request["params"];
    if (params.is_null() || !params.is_object())
    {
      const std::string response = MakeJsonRpcError(id, -32600, "Missing 'params' in resources/unsubscribe");
      SendHttpResponse(200, "application/json", response);
      return;
    }

    const JsonValue& uri_val = params["uri"];
    if (uri_val.is_null() || !uri_val.is_string())
    {
      const std::string response = MakeJsonRpcError(id, -32600, "Missing 'uri' in params");
      SendHttpResponse(200, "application/json", response);
      return;
    }

    const std::string uri(uri_val.get_string());
    auto it = std::find(s_subscribed_resources.begin(), s_subscribed_resources.end(), uri);
    if (it != s_subscribed_resources.end())
      s_subscribed_resources.erase(it);

    const std::string response = MakeJsonRpcResponse(id, "{}");
    SendHttpResponse(200, "application/json", response);
  }
  else if (method == "prompts/list")
  {
    // Static prompt definitions as raw JSON.
    static constexpr const char PROMPTS_JSON[] = R"json({"prompts":[
{"name":"debug_crash","description":"Analyze a crash by reading registers, disassembling around PC, and inspecting the stack","arguments":[{"name":"address","description":"Address to analyze (default: current PC)","required":false}]},
{"name":"analyze_memory","description":"Analyze a memory region by reading and disassembling its contents","arguments":[{"name":"address","description":"Memory address to analyze","required":true},{"name":"size","description":"Number of bytes to read (default: 256)","required":false}]},
{"name":"trace_function","description":"Disassemble a function starting at the given address","arguments":[{"name":"address","description":"Function entry point address","required":true}]},
{"name":"inspect_gpu","description":"Get a comprehensive snapshot of GPU state for debugging rendering issues","arguments":[]}
]})json";
    const std::string response = MakeJsonRpcResponse(id, PROMPTS_JSON);
    SendHttpResponse(200, "application/json", response);
  }
  else if (method == "prompts/get")
  {
    const JsonValue& params = request["params"];
    if (params.is_null() || !params.is_object())
    {
      const std::string response = MakeJsonRpcError(id, -32600, "Missing 'params' in prompts/get");
      SendHttpResponse(200, "application/json", response);
      return;
    }

    const JsonValue& name_val = params["name"];
    if (name_val.is_null() || !name_val.is_string())
    {
      const std::string response = MakeJsonRpcError(id, -32600, "Missing prompt 'name' in params");
      SendHttpResponse(200, "application/json", response);
      return;
    }

    const std::string prompt_name(name_val.get_string());
    const JsonValue& prompt_args = params["arguments"]; // may be null

    // Helper to build the final messages result with JsonWriter.
    // We'll collect the message text, then build JSON at the end.
    std::string message_text;
    bool has_message = false;

    if (prompt_name == "debug_crash")
    {
      // Get registers.
      const ToolResult reg_result = HandleReadRegisters(*JsonValue::Parse(R"({"group":"all"})"));
      std::string reg_text = reg_result.IsError() ? "{}" : reg_result.text;

      // Determine address (default: current PC).
      u32 addr = System::IsValid() ? CPU::g_state.pc : 0;
      if (!prompt_args.is_null() && prompt_args.contains("address"))
      {
        auto parsed_addr = ParseAddress(prompt_args["address"]);
        if (parsed_addr.has_value())
          addr = parsed_addr.value();
      }

      // Disassemble around PC.
      auto disasm_args = JsonValue::Parse(fmt::format(R"({{"address":{},"count":20}})", addr));
      const ToolResult disasm_result = HandleDisassemble(disasm_args.value());
      std::string disasm_text = disasm_result.IsError() ? "{}" : disasm_result.text;

      // Read stack memory (256 bytes from SP).
      u32 sp = System::IsValid() ? CPU::g_state.regs.r[29] : 0;
      auto stack_args = JsonValue::Parse(
        fmt::format(R"({{"address":{},"size":256,"format":"hex"}})", sp));
      const ToolResult stack_result = HandleReadMemory(stack_args.value());
      std::string stack_text = stack_result.IsError() ? "{}" : stack_result.text;

      message_text = fmt::format(
        "Crash analysis at address 0x{:08X}:\n\n"
        "== Registers ==\n{}\n\n"
        "== Disassembly around PC ==\n{}\n\n"
        "== Stack memory (256 bytes from SP=0x{:08X}) ==\n{}",
        addr, reg_text, disasm_text, sp, stack_text);
      has_message = true;
    }
    else if (prompt_name == "analyze_memory")
    {
      if (prompt_args.is_null() || !prompt_args.contains("address"))
      {
        const std::string response = MakeJsonRpcError(id, -32602, "Missing required argument 'address'");
        SendHttpResponse(200, "application/json", response);
        return;
      }

      auto parsed_addr = ParseAddress(prompt_args["address"]);
      if (!parsed_addr.has_value())
      {
        const std::string response = MakeJsonRpcError(id, -32602, "Invalid address");
        SendHttpResponse(200, "application/json", response);
        return;
      }

      u32 addr = parsed_addr.value();
      u32 size = 256;
      if (prompt_args.contains("size") && prompt_args["size"].is_number())
        size = static_cast<u32>(prompt_args["size"].get_uint());

      auto mem_args = JsonValue::Parse(
        fmt::format(R"({{"address":{},"size":{},"format":"hex"}})", addr, size));
      const ToolResult mem_result = HandleReadMemory(mem_args.value());
      std::string mem_text = mem_result.IsError() ? "{}" : mem_result.text;

      auto disasm_args = JsonValue::Parse(
        fmt::format(R"({{"address":{},"count":{}}})", addr, static_cast<int>(size / 4)));
      const ToolResult disasm_result = HandleDisassemble(disasm_args.value());
      std::string disasm_text = disasm_result.IsError() ? "{}" : disasm_result.text;

      message_text = fmt::format(
        "Memory analysis at 0x{:08X} ({} bytes):\n\n"
        "== Raw memory ==\n{}\n\n"
        "== Disassembly ==\n{}",
        addr, size, mem_text, disasm_text);
      has_message = true;
    }
    else if (prompt_name == "trace_function")
    {
      if (prompt_args.is_null() || !prompt_args.contains("address"))
      {
        const std::string response = MakeJsonRpcError(id, -32602, "Missing required argument 'address'");
        SendHttpResponse(200, "application/json", response);
        return;
      }

      auto parsed_addr = ParseAddress(prompt_args["address"]);
      if (!parsed_addr.has_value())
      {
        const std::string response = MakeJsonRpcError(id, -32602, "Invalid address");
        SendHttpResponse(200, "application/json", response);
        return;
      }

      u32 addr = parsed_addr.value();

      auto disasm_args = JsonValue::Parse(fmt::format(R"({{"address":{},"count":100}})", addr));
      const ToolResult disasm_result = HandleDisassemble(disasm_args.value());
      std::string disasm_text = disasm_result.IsError() ? "{}" : disasm_result.text;

      message_text = fmt::format(
        "Function trace at 0x{:08X}:\n\n"
        "== Disassembly (up to 100 instructions) ==\n{}",
        addr, disasm_text);
      has_message = true;
    }
    else if (prompt_name == "inspect_gpu")
    {
      const ToolResult gpu_result = HandleGetGpuState(*JsonValue::Parse("{}"));
      std::string gpu_text = gpu_result.IsError() ? "{}" : gpu_result.text;

      message_text = fmt::format(
        "GPU state inspection:\n\n"
        "== GPU Hardware State ==\n{}",
        gpu_text);
      has_message = true;
    }
    else
    {
      const std::string response = MakeJsonRpcError(id, -32602, fmt::format("Unknown prompt: {}", prompt_name));
      SendHttpResponse(200, "application/json", response);
      return;
    }

    // Build the result with messages array.
    JsonWriter pw;
    pw.StartObject();
    pw.Key("messages");
    pw.StartArray();
    if (has_message)
    {
      pw.StartObject();
      pw.KeyString("role", "user");
      pw.Key("content");
      pw.StartObject();
      pw.KeyString("type", "text");
      pw.KeyString("text", message_text);
      pw.EndObject();
      pw.EndObject();
    }
    pw.EndArray();
    pw.EndObject();
    const std::string response = MakeJsonRpcResponse(id, pw.GetOutput());
    SendHttpResponse(200, "application/json", response);
  }
  else if (method == "logging/setLevel")
  {
    const JsonValue& params = request["params"];
    if (params.is_null() || !params.is_object())
    {
      const std::string response = MakeJsonRpcError(id, -32600, "Missing 'params' in logging/setLevel");
      SendHttpResponse(200, "application/json", response);
      return;
    }

    const JsonValue& level_val = params["level"];
    if (level_val.is_null() || !level_val.is_string())
    {
      const std::string response = MakeJsonRpcError(id, -32602, "Missing 'level' parameter (string)");
      SendHttpResponse(200, "application/json", response);
      return;
    }

    const std::string level_str(level_val.get_string());
    Log::Level new_level;
    if (level_str == "error")
      new_level = Log::Level::Error;
    else if (level_str == "warning")
      new_level = Log::Level::Warning;
    else if (level_str == "info")
      new_level = Log::Level::Info;
    else if (level_str == "debug")
      new_level = Log::Level::Debug;
    else
    {
      const std::string response =
        MakeJsonRpcError(id, -32602, "Invalid level. Must be 'error', 'warning', 'info', or 'debug'");
      SendHttpResponse(200, "application/json", response);
      return;
    }

    s_log_stream_level = new_level;

    if (!s_log_stream_active)
    {
      Log::RegisterCallback(LogStreamCallback, nullptr);
      s_log_stream_active = true;
    }

    const std::string response = MakeJsonRpcResponse(id, "{}");
    SendHttpResponse(200, "application/json", response);
  }
  else if (method == "completion/complete")
  {
    const JsonValue& params = request["params"];
    if (params.is_null() || !params.is_object())
    {
      const std::string response = MakeJsonRpcError(id, -32600, "Missing 'params' in completion/complete");
      SendHttpResponse(200, "application/json", response);
      return;
    }

    const JsonValue& argument_val = params["argument"];
    if (argument_val.is_null() || !argument_val.is_object())
    {
      const std::string response = MakeJsonRpcError(id, -32602, "Missing 'argument' in params");
      SendHttpResponse(200, "application/json", response);
      return;
    }

    const JsonValue& arg_name_val = argument_val["name"];
    const JsonValue& arg_value_val = argument_val["value"];
    const std::string arg_name = arg_name_val.is_string() ? std::string(arg_name_val.get_string()) : "";
    const std::string arg_value = arg_value_val.is_string() ? std::string(arg_value_val.get_string()) : "";

    // Collect completion values.
    std::vector<std::string> values;

    if (arg_name == "button" || arg_name == "buttons")
    {
      // Get binding names from slot 0's controller type.
      const ControllerType type = g_settings.controller_types[0];
      if (type != ControllerType::None)
      {
        const Controller::ControllerInfo& info = Controller::GetControllerInfo(type);
        for (const auto& binding : info.bindings)
        {
          const std::string_view name(binding.name);
          if (arg_value.empty() ||
              name.substr(0, arg_value.size()) == std::string_view(arg_value))
          {
            values.push_back(std::string(binding.name));
          }
        }
      }
    }
    else if (arg_name == "feature")
    {
      static const std::array<const char*, 4> features = {
        "software_rendering", "widescreen", "pgxp", "cpu_overclock"
      };
      for (const char* f : features)
      {
        const std::string_view fv(f);
        if (arg_value.empty() ||
            fv.substr(0, arg_value.size()) == std::string_view(arg_value))
        {
          values.push_back(std::string(f));
        }
      }
    }
    else if (arg_name == "name")
    {
      // Check if a cheat-related tool is the ref.
      const JsonValue& ref_val = params["ref"];
      bool is_cheat_tool = false;
      if (!ref_val.is_null() && ref_val.is_object())
      {
        const JsonValue& ref_name_val = ref_val["name"];
        if (ref_name_val.is_string())
        {
          const std::string_view ref_name = ref_name_val.get_string();
          is_cheat_tool = (ref_name == "toggle_cheat" || ref_name == "apply_cheat");
        }
      }

      if (is_cheat_tool && System::IsValid())
      {
        const std::string serial = System::GetGameSerial();
        const GameHash hash = System::GetGameHash();
        const Cheats::CodeInfoList codes = Cheats::GetCodeInfoList(serial, hash, true, true, true);
        for (const auto& code : codes)
        {
          if (arg_value.empty() ||
              std::string_view(code.name).substr(0, arg_value.size()) ==
                std::string_view(arg_value))
          {
            values.push_back(code.name);
          }
        }
      }
    }

    // Build the completion result.
    JsonWriter cw;
    cw.StartObject();
    cw.Key("completion");
    cw.StartObject();
    cw.Key("values");
    cw.StartArray();
    for (const auto& v : values)
      cw.String(v);
    cw.EndArray();
    cw.KeyBool("hasMore", false);
    cw.KeyUint("total", static_cast<u64>(values.size()));
    cw.EndObject();
    cw.EndObject();
    const std::string response = MakeJsonRpcResponse(id, cw.GetOutput());
    SendHttpResponse(200, "application/json", response);
  }
  else
  {
    WARNING_LOG("Unknown JSON-RPC method: {}", method);
    if (is_notification)
    {
      // Notifications get no JSON-RPC response, just acknowledge the HTTP request.
      SendHttpResponse(204, "text/plain", "");
    }
    else
    {
      const std::string response = MakeJsonRpcError(id, -32601, "Method not found");
      SendHttpResponse(200, "application/json", response);
    }
  }
}

void MCPServer::ClientSocket::SendHttpResponse(int status_code, std::string_view content_type,
                                               std::string_view body)
{
  if (m_respond_via_sse)
  {
    // Streamable HTTP: SSE headers already sent on this connection.
    // Emit the response as an SSE "message" event.
    if (status_code != 204 && !body.empty())
      SendSSEEvent("message", body);
    return;
  }

  SendHttpResponseDirect(status_code, content_type, body);
}

void MCPServer::ClientSocket::SendHttpResponseDirect(int status_code, std::string_view content_type,
                                                     std::string_view body)
{
  std::string_view status_text;
  switch (status_code)
  {
    case 200:
      status_text = "OK";
      break;
    case 202:
      status_text = "Accepted";
      break;
    case 204:
      status_text = "No Content";
      break;
    case 400:
      status_text = "Bad Request";
      break;
    case 401:
      status_text = "Unauthorized";
      break;
    case 404:
      status_text = "Not Found";
      break;
    case 413:
      status_text = "Payload Too Large";
      break;
    case 503:
      status_text = "Service Unavailable";
      break;
    default:
      status_text = "Unknown";
      break;
  }

  const std::string response = fmt::format("HTTP/1.1 {} {}\r\n"
                                           "Content-Type: {}\r\n"
                                           "Content-Length: {}\r\n"
                                           "Access-Control-Allow-Origin: {}\r\n"
                                           "Connection: keep-alive\r\n"
                                           "\r\n"
                                           "{}",
                                           status_code, status_text, content_type, body.size(),
                                           s_cors_origin, body);
  if (size_t written = Write(response.data(), response.size()); written != response.size())
    ERROR_LOG("Only wrote {} of {} bytes.", written, response.size());
}

std::string MCPServer::ClientSocket::GenerateSessionId()
{
  // Generate a crypto-secure session ID per MCP spec §2.5.1.
  // 32 random hex characters = 128 bits of entropy.
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<u64> dist;
  return fmt::format("{:016x}{:016x}", dist(gen), dist(gen));
}

void MCPServer::ClientSocket::SendHttpResponseWithSessionId(int status_code, std::string_view content_type,
                                                             std::string_view body)
{
  // Same as SendHttpResponseDirect but includes MCP-Session-Id header.
  std::string_view status_text;
  switch (status_code)
  {
    case 200: status_text = "OK"; break;
    case 202: status_text = "Accepted"; break;
    default: status_text = "OK"; break;
  }

  const std::string response = fmt::format("HTTP/1.1 {} {}\r\n"
                                           "Content-Type: {}\r\n"
                                           "Content-Length: {}\r\n"
                                           "MCP-Session-Id: {}\r\n"
                                           "Access-Control-Allow-Origin: {}\r\n"
                                           "Access-Control-Expose-Headers: MCP-Session-Id\r\n"
                                           "Connection: keep-alive\r\n"
                                           "\r\n"
                                           "{}",
                                           status_code, status_text, content_type, body.size(),
                                           s_mcp_session_id, s_cors_origin, body);
  if (size_t written = Write(response.data(), response.size()); written != response.size())
    ERROR_LOG("Only wrote {} of {} bytes.", written, response.size());
}

void MCPServer::ClientSocket::SendSSEEvent(std::string_view event_name, std::string_view data_json)
{
  const std::string event = fmt::format("event: {}\ndata: {}\n\n", event_name, data_json);
  if (size_t written = Write(event.data(), event.size()); written != event.size())
    ERROR_LOG("Only wrote {} of {} SSE event bytes.", written, event.size());
}

std::string MCPServer::ClientSocket::MakeJsonRpcResponse(const JsonValue& id, std::string_view result_json)
{
  JsonWriter w;
  w.StartObject();
  w.KeyString("jsonrpc", "2.0");
  w.Key("id");
  if (id.is_null())
    w.Null();
  else if (id.is_string())
    w.String(id.get_string());
  else if (id.is_number_unsigned())
    w.Uint(id.get_uint());
  else
    w.Int(id.get_int());
  w.Key("result");
  w.RawValue(result_json);
  w.EndObject();
  return w.TakeOutput();
}

std::string MCPServer::ClientSocket::MakeJsonRpcError(const JsonValue& id, int code, std::string_view message)
{
  JsonWriter w;
  w.StartObject();
  w.KeyString("jsonrpc", "2.0");
  w.Key("id");
  if (id.is_null())
    w.Null();
  else if (id.is_string())
    w.String(id.get_string());
  else if (id.is_number_unsigned())
    w.Uint(id.get_uint());
  else
    w.Int(id.get_int());
  w.Key("error");
  w.StartObject();
  w.KeyInt("code", code);
  w.KeyString("message", message);
  w.EndObject();
  w.EndObject();
  return w.TakeOutput();
}

void MCPServer::ClientSocket::OnSystemPaused()
{
  if (m_is_sse)
  {
    const std::string pc_hex = System::IsValid() ? FormatHex32(CPU::g_state.pc) : "0x00000000";
    const std::string data = fmt::format("{{\"pc\":\"{}\",\"reason\":\"user\"}}", pc_hex);
    SendSSEEvent("system_paused", data);
  }
}

void MCPServer::ClientSocket::OnSystemResumed()
{
  if (m_is_sse)
    SendSSEEvent("system_resumed", "{}");
}

bool MCPServer::Initialize(u16 port, std::string_view auth_token, std::string_view cors_origin)
{
  s_auth_token = std::string(auth_token);
  s_cors_origin = cors_origin.empty() ? std::string("*") : std::string(cors_origin);

  Error error;
  Assert(!s_mcp_listen_socket);

  const std::optional<SocketAddress> address =
    SocketAddress::Parse(SocketAddress::Type::IPv4, "127.0.0.1", port, &error);
  if (!address.has_value())
  {
    ERROR_LOG("Failed to parse address: {}", error.GetDescription());
    return false;
  }

  SocketMultiplexer* multiplexer = System::GetSocketMultiplexer();
  if (!multiplexer)
    return false;

  s_mcp_listen_socket = multiplexer->CreateListenSocket<ClientSocket>(address.value(), &error);
  if (!s_mcp_listen_socket)
  {
    ERROR_LOG("Failed to create listen socket: {}", error.GetDescription());
    System::ReleaseSocketMultiplexer();
    return false;
  }

  INFO_LOG("MCP server is now listening on {}.", address->ToString());
  return true;
}

bool MCPServer::IsActive()
{
  return s_mcp_listen_socket != nullptr;
}

void MCPServer::Shutdown()
{
  if (!s_mcp_listen_socket)
    return;

  BroadcastSSEEvent("system_shutdown", "{}");

  INFO_LOG("Disconnecting {} MCP clients...", s_mcp_clients.size());
  while (!s_mcp_clients.empty())
  {
    // Maintain a reference so we don't delete while in scope.
    std::shared_ptr<ClientSocket> client = s_mcp_clients.back();
    client->Close();
  }

  // SSE clients are also in s_mcp_clients, so they should already be cleaned up.
  // Clear the SSE list just in case.
  s_sse_clients.clear();

  // Clear session state.
  s_mcp_session_id.clear();
  s_negotiated_protocol_version.clear();

  s_vram_watches.clear();
  s_next_vram_watch_id = 1;
  s_vram_watch_hit_pending = false;

  s_auto_releases.clear();
  s_active_sequence.reset();
  s_next_sequence_id = 1;

  if (s_log_stream_active)
  {
    Log::UnregisterCallback(LogStreamCallback, nullptr);
    s_log_stream_active = false;
  }
  s_log_stream_level = Log::Level::None;

  s_subscribed_resources.clear();
  s_state_changed_callback = nullptr;

  CleanupMCPTempFiles();
  s_memory_snapshot.clear();
  s_snapshot_base_address = 0;
  s_snapshot_size = 0;

  s_auth_token.clear();
  s_cors_origin.clear();

  INFO_LOG("Stopping MCP server.");
  s_mcp_listen_socket->Close();
  s_mcp_listen_socket.reset();
  System::ReleaseSocketMultiplexer();
}

void MCPServer::OnSystemPaused()
{
  // If a VRAM watch triggered the pause, send the SSE notification now.
  if (s_vram_watch_hit_pending)
  {
    JsonWriter ew;
    ew.StartObject();
    ew.KeyString("type", "vram_write");
    ew.KeyUint("x", s_vram_watch_hit_x);
    ew.KeyUint("y", s_vram_watch_hit_y);
    ew.KeyUint("width", s_vram_watch_hit_w);
    ew.KeyUint("height", s_vram_watch_hit_h);
    ew.KeyString("pc", fmt::format("0x{:08X}", s_vram_watch_hit_pc));
    ew.EndObject();
    BroadcastSSEEvent("vram_watch_hit", ew.GetOutput());
    // Keep s_vram_watch_hit_pending true so get_vram_watch_last_hit can read it.
  }

  for (auto& it : s_sse_clients)
    it->OnSystemPaused();

  NotifyResourceUpdated("emulator://status");
  NotifyResourceUpdated("emulator://registers");
}

void MCPServer::OnSystemResumed()
{
  for (auto& it : s_sse_clients)
    it->OnSystemResumed();

  NotifyResourceUpdated("emulator://status");
}

bool MCPServer::OnVRAMWrite(u16 x, u16 y, u16 width, u16 height)
{
  if (s_vram_watches.empty())
    return false;

  // Check if the VRAM write rectangle intersects any watched region.
  for (const auto& watch : s_vram_watches)
  {
    const bool intersects = (x < watch.x + watch.width) && (x + width > watch.x) &&
                            (y < watch.y + watch.height) && (y + height > watch.y);
    if (intersects)
    {
      DEV_LOG("VRAM watch #{} hit: write ({},{}) {}x{} intersects watch ({},{}) {}x{}",
              watch.id, x, y, width, height, watch.x, watch.y, watch.width, watch.height);

      // Store hit info for the SSE notification (sent when system actually pauses).
      s_vram_watch_hit_pending = true;
      s_vram_watch_hit_x = x;
      s_vram_watch_hit_y = y;
      s_vram_watch_hit_w = width;
      s_vram_watch_hit_h = height;
      s_vram_watch_hit_pc = CPU::g_state.pc;
      s_vram_watch_hit_ra = CPU::g_state.regs.r[31]; // ra
      s_vram_watch_hit_sp = CPU::g_state.regs.r[29]; // sp

      // Snapshot all 32 GPRs.
      std::memcpy(s_vram_watch_hit_regs.data(), CPU::g_state.regs.r, 32 * sizeof(u32));

      // Snapshot 256 bytes of stack from SP (physical address).
      const u32 sp_phys = CPU::g_state.regs.r[29] & 0x1FFFFF;
      for (u32 i = 0; i < 64; i++)
      {
        const u32 addr = (sp_phys + i * 4) & 0x1FFFFF;
        std::memcpy(&s_vram_watch_hit_stack[i], Bus::g_ram + addr, sizeof(u32));
      }

      INFO_LOG("VRAM watch hit at PC=0x{:08X} RA=0x{:08X} SP=0x{:08X}, VRAM write ({},{}) {}x{}",
               CPU::g_state.pc, CPU::g_state.regs.r[31], CPU::g_state.regs.r[29], x, y, width, height);

      // Request system pause.
      System::PauseSystem(true);
      return true;
    }
  }

  return false;
}

void MCPServer::OnFrameEnd()
{
  if (!s_mcp_listen_socket)
    return;

  // Tick auto-release timers.
  for (auto it = s_auto_releases.begin(); it != s_auto_releases.end();)
  {
    if (--it->frames_remaining == 0)
    {
      Controller* controller = GetControllerForSlot(it->slot);
      if (controller)
        controller->SetBindState(it->bind_index, 0.0f);
      it = s_auto_releases.erase(it);
    }
    else
    {
      ++it;
    }
  }

  // Tick active input sequence.
  if (s_active_sequence.has_value())
  {
    auto& seq = s_active_sequence.value();
    seq.frames_in_step++;

    if (seq.frames_in_step >= seq.steps[seq.current_step].duration_frames)
    {
      Controller* controller = GetControllerForSlot(seq.slot);
      if (controller)
      {
        for (u32 idx : seq.steps[seq.current_step].bind_indices)
          controller->SetBindState(idx, 0.0f);
      }

      seq.current_step++;
      seq.frames_in_step = 0;

      if (seq.current_step >= static_cast<u32>(seq.steps.size()))
      {
        const std::string data =
          fmt::format("{{\"sequence_id\":{},\"total_steps\":{}}}", seq.id, seq.steps.size());
        BroadcastSSEEvent("input_sequence_complete", data);
        s_active_sequence.reset();
      }
      else if (controller)
      {
        for (u32 idx : seq.steps[seq.current_step].bind_indices)
          controller->SetBindState(idx, 1.0f);
      }
    }
  }
}
