#ifndef INCLUDED_GL_UTILS
#define INCLUDED_GL_UTILS

#pragma once

#include <GL3/gl3w.h>
#include <GL/glu.h>
#undef ERROR

#include <stdexcept>
#include <string>
#include <sstream>

namespace
{
  std::string genErrorString(GLenum err)
  {
  	if (const char* str = reinterpret_cast<const char*>(gluErrorString(err)))
  		return str + '\n';
  	return "unknown error code\n";
  }
  
  std::string genErrorString(GLenum err, const char* file, int line)
  {
  	std::ostringstream msg;
  	msg << file << '(' << line << "): error: ";
  
  	if (const char* str = reinterpret_cast<const char*>(gluErrorString(err)))
  		msg << str;
  	else
  		msg << "unknown error code\n";
  	return msg.str();
  }
}

namespace GL
{
  class error : public std::runtime_error
  {
  public:
    error(GLenum err, const char* file, int line) : runtime_error(genErrorString(err, file, line))
    {
    }

    error(GLenum err) : runtime_error(genErrorString(err))
    {
    }

    error(const std::string& msg) : runtime_error(msg)
    {
    }
  };

  void checkError(GLenum error, const char* file, int line)
  {
    if (error)
    {
#if defined(_WIN32) && defined(_DEBUG)
      OutputDebugStringA(genErrorString(error).c_str());
      //DebugBreak();
#else
      throw GL::error(error, file, line);
#endif
    }
  }

  void checkError(const char* file, int line)
  {
    checkError(glGetError(), file, line);
  }

  void checkError()
  {
    if (GLenum error = glGetError())
    {
#if defined(_WIN32) && defined(_DEBUG)
      OutputDebugStringA(genErrorString(error).c_str());
      DebugBreak();
#else
      throw GL::error(error);
#endif
    }
  }

  template <GLenum type>
  struct GLToType;

  template <>
  struct GLToType<GL_BYTE>
  {
    static const GLenum type = GL_BYTE;
    static const int size = 1;
    typedef GLbyte T;
  };

  template <>
  struct GLToType<GL_UNSIGNED_BYTE>
  {
    static const GLenum type = GL_UNSIGNED_BYTE;
    static const int size = 1;
    typedef GLubyte T;
  };

  template <>
  struct GLToType<GL_SHORT>
  {
    static const GLenum type = GL_SHORT;
    static const int size = 2;
    typedef GLshort T;
  };

  template <>
  struct GLToType<GL_UNSIGNED_SHORT>
  {
    static const GLenum type = GL_UNSIGNED_SHORT;
    static const int size = 2;
    typedef GLushort T;
  };

  template <>
  struct GLToType<GL_INT>
  {
    static const GLenum type = GL_INT;
    static const int size = 4;
    typedef GLint T;
  };

  template <>
  struct GLToType<GL_UNSIGNED_INT>
  {
    static const GLenum type = GL_UNSIGNED_INT;
    static const int size = 4;
    typedef GLuint T;
  };

  template <>
  struct GLToType<GL_FLOAT>
  {
    static const GLenum type = GL_FLOAT;
    static const int size = 4;
    typedef GLfloat T;
  };


  template <typename T>
  struct TypeToGL;

  template <>
  struct TypeToGL<GLbyte>
  {
    static const GLenum type = GL_BYTE;
    static const int size = 1;
    typedef GLbyte T;
  };

  template <>
  struct TypeToGL<GLubyte>
  {
    static const GLenum type = GL_UNSIGNED_BYTE;
    static const int size = 1;
    typedef GLubyte T;
  };

  template <>
  struct TypeToGL<GLshort>
  {
    static const GLenum type = GL_SHORT;
    static const int size = 2;
    typedef GLshort T;
  };

  template <>
  struct TypeToGL<GLushort>
  {
    static const GLenum type = GL_UNSIGNED_SHORT;
    static const int size = 2;
    typedef GLushort T;
  };

  template <>
  struct TypeToGL<GLint>
  {
    static const GLenum type = GL_INT;
    static const int size = 4;
    typedef GLint T;
  };

  template <>
  struct TypeToGL<GLuint>
  {
    static const GLenum type = GL_UNSIGNED_INT;
    static const int size = 4;
    typedef GLuint T;
  };

  template <>
  struct TypeToGL<GLfloat>
  {
    static const GLenum type = GL_FLOAT;
    static const int size = 4;
    typedef GLfloat T;
  };
}

#define GL_CHECK_ERROR() GL::checkError(__FILE__, __LINE__)

#endif  // INCLUDED_GL_UTILS
