// SPDX-FileCopyrightText: 2019-2026 Connor McLaughlin <stenzek@gmail.com> and contributors.
// SPDX-License-Identifier: CC-BY-NC-ND-4.0

#pragma once

#include "types.h"

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

class JsonParser;

class JsonValue
{
  friend class JsonParser;

public:
  enum class Type : u8
  {
    Null,
    Bool,
    SignedInt,
    UnsignedInt,
    Float,
    String,
    Array,
    Object,
  };

  using ObjectEntry = std::pair<std::string, JsonValue>;
  using ArrayStorage = std::vector<JsonValue>;
  using ObjectStorage = std::vector<ObjectEntry>;

  JsonValue();
  ~JsonValue();
  JsonValue(const JsonValue& other);
  JsonValue(JsonValue&& other) noexcept;
  JsonValue& operator=(const JsonValue& other);
  JsonValue& operator=(JsonValue&& other) noexcept;

  Type GetType() const { return m_type; }

  bool is_null() const { return m_type == Type::Null; }
  bool is_bool() const { return m_type == Type::Bool; }
  bool is_string() const { return m_type == Type::String; }
  bool is_array() const { return m_type == Type::Array; }
  bool is_object() const { return m_type == Type::Object; }
  bool is_number() const;
  bool is_number_unsigned() const { return m_type == Type::UnsignedInt; }
  bool is_number_integer() const { return m_type == Type::SignedInt || m_type == Type::UnsignedInt; }

  bool get_bool() const;
  s64 get_int() const;
  u64 get_uint() const;
  double get_float() const;
  std::string_view get_string() const;

  size_t size() const;
  bool contains(std::string_view key) const;

  const JsonValue& operator[](std::string_view key) const;
  const JsonValue& operator[](size_t index) const;

  // Array iteration.
  const JsonValue* begin() const;
  const JsonValue* end() const;

  // Object iteration.
  const ObjectStorage& get_object() const;

  static std::optional<JsonValue> Parse(std::string_view input);

  static const JsonValue NULL_VALUE;

private:
  void Destroy();

  Type m_type = Type::Null;

  union
  {
    bool m_bool;
    s64 m_int;
    u64 m_uint;
    double m_float;
  };

  std::string m_string;
  ArrayStorage m_array;
  ObjectStorage m_object;
};
