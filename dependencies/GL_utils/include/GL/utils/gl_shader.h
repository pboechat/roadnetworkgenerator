#ifndef INCLUDED_GL_UTILS_SHADER
#define INCLUDED_GL_UTILS_SHADER

#pragma once

#include <utility>
#include <string>

#include "gl.h"

#include <stdexcept>
#include <memory>
#include <string>

namespace GL
{
  bool shaderCompileStatus(GLuint shader)
  {
    GLint b;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &b);
    GL_CHECK_ERROR();
    return b == GL_TRUE;
  }

  std::string shaderInfoLog(GLuint shader)
  {
    GLint length;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
    GL_CHECK_ERROR();
    if (length == 0)
      return "";
    std::unique_ptr<char[]> log(new char[length + 1]);
    glGetShaderInfoLog(shader, length, 0, reinterpret_cast<GLchar*>(log.get()));
    GL_CHECK_ERROR();
    return log.get();
  }

  template <GLenum ShaderType>
  class Shader
  {
  private:
    Shader(const Shader&);
    Shader& operator =(const Shader&);

    GLuint shader;

  public:
    static const GLenum type = ShaderType;

    Shader(const char* source)
      : shader(glCreateShader(ShaderType))
    {
      glShaderSource(shader, 1, &source, 0);
      glCompileShader(shader);
      GL_CHECK_ERROR();
      if (shaderCompileStatus(shader) == false)
        throw error("error compiling Shader: " + shaderInfoLog(shader));
    }

    Shader(Shader&& s)
      : shader(s.shader)
    {
      s.shader = 0;
    }

    ~Shader()
    {
      glDeleteShader(shader);
    }

    Shader& operator =(Shader&& s)
    {
      using std::swap;
      swap(shader, s.shader);
      return *this;
    }

    void swap(Shader& b)
    {
      using std::swap;
      swap(shader, b.shader);
    }

    operator GLuint() const { return shader; }

    friend bool compileStatus(const Shader& shader)
    {
      return shaderCompileStatus(shader);
    }

    friend std::string infoLog(const Shader& shader)
    {
      return shaderInfoLog(shader);
    }

    friend void swap(GL::Shader<ShaderType>& a, GL::Shader<ShaderType>& b)
    {
      a.swap(b);
    }
  };

  typedef Shader<GL_VERTEX_SHADER> VertexShader;
  typedef Shader<GL_GEOMETRY_SHADER> GeometryShader;
  typedef Shader<GL_FRAGMENT_SHADER> FragmentShader;

  bool programLinkStatus(GLuint program)
  {
    GLint b;
    glGetProgramiv(program, GL_LINK_STATUS, &b);
    GL_CHECK_ERROR();
    return b == GL_TRUE;
  }

  bool programValidateStatus(GLuint program)
  {
    GLint b;
    glGetProgramiv(program, GL_VALIDATE_STATUS, &b);
    GL_CHECK_ERROR();
    return b == GL_TRUE;
  }

  std::string programInfoLog(GLuint program)
  {
    GLint length;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
    GL_CHECK_ERROR();
    if (length == 0)
      return "";
    std::unique_ptr<char[]> log(new char[length + 1]);
    glGetProgramInfoLog(program, length, 0, reinterpret_cast<char*>(log.get()));
    GL_CHECK_ERROR();
    return log.get();
  }

  class Program
  {
  private:
    Program(const Program&);
    Program& operator =(const Program&);

    GLuint program;

  public:
    Program()
    : program(glCreateProgram())
    {
      GL_CHECK_ERROR();
    }

    Program(Program&& p)
      : program(p.program)
    {
      p.program = 0;
    }

    ~Program()
    {
      glDeleteProgram(program);
    }

    Program& operator =(Program&& p)
    {
      using std::swap;
      swap(program, p.program);
      return *this;
    }

    void swap(Program& b)
    {
      using std::swap;
      swap(program, b.program);
    }

    operator GLuint() const { return program; }

    template <class ShaderType>
    void attachShader(const ShaderType& shader)
    {
      glAttachShader(program, shader);
      GL_CHECK_ERROR();
    }

    template <class ShaderType>
    void detachShader(const ShaderType& shader)
    {
      glDetachShader(program, shader);
      GL_CHECK_ERROR();
    }

    void link()
    {
      glLinkProgram(program);
      GL_CHECK_ERROR();
      if (programLinkStatus(program) == false)
        throw error("error linking program: " + programInfoLog(program));
    }

    friend bool linkStatus(const Program& program)
    {
      return programLinkStatus(program);
    }

    friend bool validate(const Program& program)
    {
      return programValidateStatus(program);
    }

    friend std::string infoLog(const Program& program)
    {
      return programInfoLog(program);
    }
  };

  using std::swap;
  using std::move;
  using std::forward;
}

namespace std
{
  template <>
  inline void swap<GL::Program>(GL::Program& a, GL::Program& b)
  {
    a.swap(b);
  }
}

#endif  // INCLUDED_GL_UTILS_SHADER
