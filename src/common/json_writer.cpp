// SPDX-FileCopyrightText: 2019-2026 Connor McLaughlin <stenzek@gmail.com> and contributors.
// SPDX-License-Identifier: CC-BY-NC-ND-4.0

#include "json_writer.h"

#include "fmt/format.h"

#include <iterator>

JsonWriter::JsonWriter()
{
  m_buffer.reserve(256);
}

void JsonWriter::WriteSeparator()
{
  if (m_needs_comma)
    m_buffer += ',';
  m_needs_comma = true;
}

void JsonWriter::WriteQuotedString(std::string_view str)
{
  m_buffer += '"';
  EscapeString(m_buffer, str);
  m_buffer += '"';
}

void JsonWriter::StartObject()
{
  WriteSeparator();
  m_buffer += '{';
  m_needs_comma = false;
}

void JsonWriter::EndObject()
{
  m_buffer += '}';
  m_needs_comma = true;
}

void JsonWriter::StartArray()
{
  WriteSeparator();
  m_buffer += '[';
  m_needs_comma = false;
}

void JsonWriter::EndArray()
{
  m_buffer += ']';
  m_needs_comma = true;
}

void JsonWriter::Key(std::string_view key)
{
  WriteSeparator();
  WriteQuotedString(key);
  m_buffer += ':';
  m_needs_comma = false;
}

void JsonWriter::Null()
{
  WriteSeparator();
  m_buffer += "null";
}

void JsonWriter::Bool(bool val)
{
  WriteSeparator();
  m_buffer += val ? "true" : "false";
}

void JsonWriter::Int(s64 val)
{
  WriteSeparator();
  fmt::format_to(std::back_inserter(m_buffer), "{}", val);
}

void JsonWriter::Uint(u64 val)
{
  WriteSeparator();
  fmt::format_to(std::back_inserter(m_buffer), "{}", val);
}

void JsonWriter::Double(double val)
{
  WriteSeparator();
  fmt::format_to(std::back_inserter(m_buffer), "{}", val);
}

void JsonWriter::String(std::string_view val)
{
  WriteSeparator();
  WriteQuotedString(val);
}

void JsonWriter::RawValue(std::string_view json)
{
  WriteSeparator();
  m_buffer.append(json);
}

void JsonWriter::KeyNull(std::string_view key)
{
  Key(key);
  m_buffer += "null";
  m_needs_comma = true;
}

void JsonWriter::KeyBool(std::string_view key, bool val)
{
  Key(key);
  m_buffer += val ? "true" : "false";
  m_needs_comma = true;
}

void JsonWriter::KeyInt(std::string_view key, s64 val)
{
  Key(key);
  fmt::format_to(std::back_inserter(m_buffer), "{}", val);
  m_needs_comma = true;
}

void JsonWriter::KeyUint(std::string_view key, u64 val)
{
  Key(key);
  fmt::format_to(std::back_inserter(m_buffer), "{}", val);
  m_needs_comma = true;
}

void JsonWriter::KeyDouble(std::string_view key, double val)
{
  Key(key);
  fmt::format_to(std::back_inserter(m_buffer), "{}", val);
  m_needs_comma = true;
}

void JsonWriter::KeyString(std::string_view key, std::string_view val)
{
  Key(key);
  WriteQuotedString(val);
  m_needs_comma = true;
}

void JsonWriter::KeyRawValue(std::string_view key, std::string_view json)
{
  Key(key);
  m_buffer.append(json);
  m_needs_comma = true;
}

std::string_view JsonWriter::GetOutput() const
{
  return m_buffer;
}

std::string JsonWriter::TakeOutput()
{
  return std::move(m_buffer);
}

void JsonWriter::EscapeString(std::string& out, std::string_view str)
{
  for (const char c : str)
  {
    switch (c)
    {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      case '\b':
        out += "\\b";
        break;
      case '\f':
        out += "\\f";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20)
          fmt::format_to(std::back_inserter(out), "\\u{:04x}", static_cast<unsigned>(static_cast<unsigned char>(c)));
        else
          out += c;
        break;
    }
  }
}
