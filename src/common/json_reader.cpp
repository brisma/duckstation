// SPDX-FileCopyrightText: 2019-2026 Connor McLaughlin <stenzek@gmail.com> and contributors.
// SPDX-License-Identifier: CC-BY-NC-ND-4.0

#include "json_reader.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

const JsonValue JsonValue::NULL_VALUE;

JsonValue::JsonValue() : m_type(Type::Null), m_bool(false)
{
}

JsonValue::~JsonValue()
{
  Destroy();
}

JsonValue::JsonValue(const JsonValue& other) : m_type(other.m_type), m_string(other.m_string),
  m_array(other.m_array), m_object(other.m_object)
{
  switch (m_type)
  {
    case Type::Bool: m_bool = other.m_bool; break;
    case Type::SignedInt: m_int = other.m_int; break;
    case Type::UnsignedInt: m_uint = other.m_uint; break;
    case Type::Float: m_float = other.m_float; break;
    default: break;
  }
}

JsonValue::JsonValue(JsonValue&& other) noexcept : m_type(other.m_type), m_string(std::move(other.m_string)),
  m_array(std::move(other.m_array)), m_object(std::move(other.m_object))
{
  switch (m_type)
  {
    case Type::Bool: m_bool = other.m_bool; break;
    case Type::SignedInt: m_int = other.m_int; break;
    case Type::UnsignedInt: m_uint = other.m_uint; break;
    case Type::Float: m_float = other.m_float; break;
    default: break;
  }
  other.m_type = Type::Null;
}

JsonValue& JsonValue::operator=(const JsonValue& other)
{
  if (this != &other)
  {
    Destroy();
    m_type = other.m_type;
    m_string = other.m_string;
    m_array = other.m_array;
    m_object = other.m_object;
    switch (m_type)
    {
      case Type::Bool: m_bool = other.m_bool; break;
      case Type::SignedInt: m_int = other.m_int; break;
      case Type::UnsignedInt: m_uint = other.m_uint; break;
      case Type::Float: m_float = other.m_float; break;
      default: break;
    }
  }
  return *this;
}

JsonValue& JsonValue::operator=(JsonValue&& other) noexcept
{
  if (this != &other)
  {
    Destroy();
    m_type = other.m_type;
    m_string = std::move(other.m_string);
    m_array = std::move(other.m_array);
    m_object = std::move(other.m_object);
    switch (m_type)
    {
      case Type::Bool: m_bool = other.m_bool; break;
      case Type::SignedInt: m_int = other.m_int; break;
      case Type::UnsignedInt: m_uint = other.m_uint; break;
      case Type::Float: m_float = other.m_float; break;
      default: break;
    }
    other.m_type = Type::Null;
  }
  return *this;
}

void JsonValue::Destroy()
{
  m_string.clear();
  m_array.clear();
  m_object.clear();
  m_type = Type::Null;
}

bool JsonValue::is_number() const
{
  return m_type == Type::SignedInt || m_type == Type::UnsignedInt || m_type == Type::Float;
}

bool JsonValue::get_bool() const
{
  return (m_type == Type::Bool) ? m_bool : false;
}

s64 JsonValue::get_int() const
{
  switch (m_type)
  {
    case Type::SignedInt: return m_int;
    case Type::UnsignedInt: return static_cast<s64>(m_uint);
    case Type::Float: return static_cast<s64>(m_float);
    default: return 0;
  }
}

u64 JsonValue::get_uint() const
{
  switch (m_type)
  {
    case Type::UnsignedInt: return m_uint;
    case Type::SignedInt: return static_cast<u64>(m_int);
    case Type::Float: return static_cast<u64>(m_float);
    default: return 0;
  }
}

double JsonValue::get_float() const
{
  switch (m_type)
  {
    case Type::Float: return m_float;
    case Type::SignedInt: return static_cast<double>(m_int);
    case Type::UnsignedInt: return static_cast<double>(m_uint);
    default: return 0.0;
  }
}

std::string_view JsonValue::get_string() const
{
  return (m_type == Type::String) ? std::string_view(m_string) : std::string_view();
}

size_t JsonValue::size() const
{
  switch (m_type)
  {
    case Type::Array: return m_array.size();
    case Type::Object: return m_object.size();
    default: return 0;
  }
}

bool JsonValue::contains(std::string_view key) const
{
  if (m_type != Type::Object)
    return false;
  for (const auto& entry : m_object)
  {
    if (entry.first == key)
      return true;
  }
  return false;
}

const JsonValue& JsonValue::operator[](std::string_view key) const
{
  if (m_type == Type::Object)
  {
    for (const auto& entry : m_object)
    {
      if (entry.first == key)
        return entry.second;
    }
  }
  return NULL_VALUE;
}

const JsonValue& JsonValue::operator[](size_t index) const
{
  if (m_type == Type::Array && index < m_array.size())
    return m_array[index];
  return NULL_VALUE;
}

const JsonValue* JsonValue::begin() const
{
  return (m_type == Type::Array && !m_array.empty()) ? m_array.data() : nullptr;
}

const JsonValue* JsonValue::end() const
{
  return (m_type == Type::Array && !m_array.empty()) ? m_array.data() + m_array.size() : nullptr;
}

const JsonValue::ObjectStorage& JsonValue::get_object() const
{
  static const ObjectStorage s_empty;
  return (m_type == Type::Object) ? m_object : s_empty;
}

// ---- Parser ----

class JsonParser
{
public:
  JsonParser(std::string_view input) : m_input(input), m_pos(0) {}

  std::optional<JsonValue> Parse()
  {
    SkipWhitespace();
    auto val = ParseValue();
    if (!val.has_value())
      return std::nullopt;
    SkipWhitespace();
    return val;
  }

private:
  void SkipWhitespace()
  {
    while (m_pos < m_input.size() && (m_input[m_pos] == ' ' || m_input[m_pos] == '\t' ||
                                       m_input[m_pos] == '\n' || m_input[m_pos] == '\r'))
      m_pos++;
  }

  char Peek() const { return (m_pos < m_input.size()) ? m_input[m_pos] : '\0'; }
  char Next() { return (m_pos < m_input.size()) ? m_input[m_pos++] : '\0'; }

  bool Expect(char c)
  {
    SkipWhitespace();
    if (Peek() == c)
    {
      m_pos++;
      return true;
    }
    return false;
  }

  bool MatchLiteral(const char* lit)
  {
    const size_t len = std::strlen(lit);
    if (m_pos + len > m_input.size())
      return false;
    if (m_input.substr(m_pos, len) == std::string_view(lit, len))
    {
      m_pos += len;
      return true;
    }
    return false;
  }

  std::optional<JsonValue> ParseValue()
  {
    SkipWhitespace();
    const char c = Peek();

    if (c == '"')
      return ParseString();
    if (c == '{')
      return ParseObject();
    if (c == '[')
      return ParseArray();
    if (c == 't' || c == 'f')
      return ParseBool();
    if (c == 'n')
      return ParseNull();
    if (c == '-' || (c >= '0' && c <= '9'))
      return ParseNumber();

    return std::nullopt;
  }

  std::optional<JsonValue> ParseNull()
  {
    if (!MatchLiteral("null"))
      return std::nullopt;
    JsonValue v;
    return v;
  }

  std::optional<JsonValue> ParseBool()
  {
    JsonValue v;
    v.m_type = JsonValue::Type::Bool;
    if (MatchLiteral("true"))
    {
      v.m_bool = true;
      return v;
    }
    if (MatchLiteral("false"))
    {
      v.m_bool = false;
      return v;
    }
    return std::nullopt;
  }

  std::optional<JsonValue> ParseNumber()
  {
    const size_t start = m_pos;
    bool is_negative = false;
    bool has_dot = false;
    bool has_exp = false;

    if (Peek() == '-')
    {
      is_negative = true;
      m_pos++;
    }

    while (m_pos < m_input.size())
    {
      const char c = m_input[m_pos];
      if (c >= '0' && c <= '9')
      {
        m_pos++;
      }
      else if (c == '.' && !has_dot)
      {
        has_dot = true;
        m_pos++;
      }
      else if ((c == 'e' || c == 'E') && !has_exp)
      {
        has_exp = true;
        m_pos++;
        if (m_pos < m_input.size() && (m_input[m_pos] == '+' || m_input[m_pos] == '-'))
          m_pos++;
      }
      else
      {
        break;
      }
    }

    const std::string_view num_str = m_input.substr(start, m_pos - start);
    if (num_str.empty())
      return std::nullopt;

    JsonValue v;

    if (has_dot || has_exp)
    {
      char* endptr = nullptr;
      // Need null-terminated string for strtod.
      char buf[64];
      const size_t len = std::min(num_str.size(), sizeof(buf) - 1);
      std::memcpy(buf, num_str.data(), len);
      buf[len] = '\0';
      v.m_float = std::strtod(buf, &endptr);
      v.m_type = JsonValue::Type::Float;
    }
    else if (is_negative)
    {
      char* endptr = nullptr;
      char buf[64];
      const size_t len = std::min(num_str.size(), sizeof(buf) - 1);
      std::memcpy(buf, num_str.data(), len);
      buf[len] = '\0';
      v.m_int = std::strtoll(buf, &endptr, 10);
      v.m_type = JsonValue::Type::SignedInt;
    }
    else
    {
      char* endptr = nullptr;
      char buf[64];
      const size_t len = std::min(num_str.size(), sizeof(buf) - 1);
      std::memcpy(buf, num_str.data(), len);
      buf[len] = '\0';
      v.m_uint = std::strtoull(buf, &endptr, 10);
      v.m_type = JsonValue::Type::UnsignedInt;
    }

    return v;
  }

  std::optional<JsonValue> ParseString()
  {
    if (Next() != '"')
      return std::nullopt;

    std::string result;
    while (m_pos < m_input.size())
    {
      char c = m_input[m_pos++];
      if (c == '"')
      {
        JsonValue v;
        v.m_type = JsonValue::Type::String;
        v.m_string = std::move(result);
        return v;
      }
      if (c == '\\')
      {
        if (m_pos >= m_input.size())
          return std::nullopt;
        c = m_input[m_pos++];
        switch (c)
        {
          case '"': result += '"'; break;
          case '\\': result += '\\'; break;
          case '/': result += '/'; break;
          case 'n': result += '\n'; break;
          case 'r': result += '\r'; break;
          case 't': result += '\t'; break;
          case 'b': result += '\b'; break;
          case 'f': result += '\f'; break;
          case 'u':
          {
            if (m_pos + 4 > m_input.size())
              return std::nullopt;
            char hex[5] = {};
            std::memcpy(hex, &m_input[m_pos], 4);
            m_pos += 4;
            const unsigned codepoint = static_cast<unsigned>(std::strtoul(hex, nullptr, 16));
            if (codepoint < 0x80)
            {
              result += static_cast<char>(codepoint);
            }
            else if (codepoint < 0x800)
            {
              result += static_cast<char>(0xC0 | (codepoint >> 6));
              result += static_cast<char>(0x80 | (codepoint & 0x3F));
            }
            else
            {
              result += static_cast<char>(0xE0 | (codepoint >> 12));
              result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
              result += static_cast<char>(0x80 | (codepoint & 0x3F));
            }
            break;
          }
          default:
            return std::nullopt;
        }
      }
      else
      {
        result += c;
      }
    }
    return std::nullopt; // unterminated string
  }

  std::optional<JsonValue> ParseArray()
  {
    if (Next() != '[')
      return std::nullopt;

    JsonValue v;
    v.m_type = JsonValue::Type::Array;

    SkipWhitespace();
    if (Peek() == ']')
    {
      m_pos++;
      return v;
    }

    for (;;)
    {
      auto elem = ParseValue();
      if (!elem.has_value())
        return std::nullopt;
      v.m_array.push_back(std::move(elem.value()));

      SkipWhitespace();
      if (Peek() == ']')
      {
        m_pos++;
        return v;
      }
      if (!Expect(','))
        return std::nullopt;
    }
  }

  std::optional<JsonValue> ParseObject()
  {
    if (Next() != '{')
      return std::nullopt;

    JsonValue v;
    v.m_type = JsonValue::Type::Object;

    SkipWhitespace();
    if (Peek() == '}')
    {
      m_pos++;
      return v;
    }

    for (;;)
    {
      SkipWhitespace();
      auto key = ParseString();
      if (!key.has_value() || key->m_type != JsonValue::Type::String)
        return std::nullopt;

      if (!Expect(':'))
        return std::nullopt;

      auto val = ParseValue();
      if (!val.has_value())
        return std::nullopt;

      v.m_object.emplace_back(std::move(key->m_string), std::move(val.value()));

      SkipWhitespace();
      if (Peek() == '}')
      {
        m_pos++;
        return v;
      }
      if (!Expect(','))
        return std::nullopt;
    }
  }

  std::string_view m_input;
  size_t m_pos;
};

std::optional<JsonValue> JsonValue::Parse(std::string_view input)
{
  JsonParser parser(input);
  return parser.Parse();
}
