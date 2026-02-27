// SPDX-FileCopyrightText: 2019-2026 Connor McLaughlin <stenzek@gmail.com> and contributors.
// SPDX-License-Identifier: CC-BY-NC-ND-4.0

#pragma once

#include "types.h"

#include <string>
#include <string_view>

class JsonWriter
{
public:
  JsonWriter();

  void StartObject();
  void EndObject();
  void StartArray();
  void EndArray();

  void Key(std::string_view key);

  void Null();
  void Bool(bool val);
  void Int(s64 val);
  void Uint(u64 val);
  void Double(double val);
  void String(std::string_view val);
  void RawValue(std::string_view json);

  void KeyNull(std::string_view key);
  void KeyBool(std::string_view key, bool val);
  void KeyInt(std::string_view key, s64 val);
  void KeyUint(std::string_view key, u64 val);
  void KeyDouble(std::string_view key, double val);
  void KeyString(std::string_view key, std::string_view val);
  void KeyRawValue(std::string_view key, std::string_view json);

  std::string_view GetOutput() const;
  std::string TakeOutput();

  static void EscapeString(std::string& out, std::string_view str);

private:
  void WriteSeparator();
  void WriteQuotedString(std::string_view str);

  std::string m_buffer;
  bool m_needs_comma = false;
};
