// SPDX-FileCopyrightText: 2019-2026 Connor McLaughlin <stenzek@gmail.com> and contributors.
// SPDX-License-Identifier: CC-BY-NC-ND-4.0

#pragma once
#include <functional>
#include <string_view>

namespace MCPServer {

/// Callback invoked on the UI thread after any MCP tool mutates system state.
/// The frontend registers this to refresh debugger views (breakpoints, registers, etc.).
using StateChangedCallback = std::function<void()>;
void SetStateChangedCallback(StateChangedCallback callback);
bool Initialize(u16 port, std::string_view auth_token = {}, std::string_view cors_origin = {});
void Shutdown();

/// Returns true if the MCP server listen socket is active.
bool IsActive();

void OnSystemPaused();
void OnSystemResumed();

/// Called from GPU::FinishVRAMWrite() to check VRAM write watchpoints.
/// Returns true if a watchpoint was hit and the system should pause.
bool OnVRAMWrite(u16 x, u16 y, u16 width, u16 height);

/// Called from System::FrameDone() to tick auto-release timers and input sequences.
void OnFrameEnd();
} // namespace MCPServer
