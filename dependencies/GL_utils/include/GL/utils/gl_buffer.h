#ifndef INCLUDED_GL_UTILS_BUFFER
#define INCLUDED_GL_UTILS_BUFFER

#pragma once

#include <utility>

#include "gl.h"


namespace GL
{

  class VertexArrayObject
  {
  private:
    VertexArrayObject(const VertexArrayObject&);
    VertexArrayObject& operator =(const VertexArrayObject&);

    static GLuint create()
    {
      GLuint id;
      glGenVertexArrays(1, &id);
      checkError();
      if (id == 0)
        throw error("couldn't create Vertex Array Object");
      return id;
    }

    GLuint id;
  public:
    VertexArrayObject()
      : id(create())
    {
    }

    VertexArrayObject(GLuint id)
      : id(id)
    {
    }

    VertexArrayObject(VertexArrayObject&& vao)
      : id(vao.id)
    {
      vao.id = 0;
    }

    ~VertexArrayObject()
    {
      glDeleteVertexArrays(1, &id);
    }

    VertexArrayObject& operator =(VertexArrayObject&& vao)
    {
      using std::swap;
      swap(id, vao.id);
      return *this;
    }

    void swap(VertexArrayObject& vao)
    {
      using std::swap;
      swap(id, vao.id);
    }

    operator GLuint() const { return id; }
  };

  class Buffer
  {
  private:
    Buffer(const Buffer& b);
    Buffer& operator =(const Buffer& b);

    GLuint id;

    static GLuint create()
    {
      GLuint id;
      glGenBuffers(1, &id);
      checkError();
      if (id == 0)
        throw error("couldn't create Buffer Object");
      return id;
    }

  public:
    Buffer()
      : id(create())
    {
    }

    Buffer(GLuint id)
      : id(id)
    {
    }

    Buffer(Buffer&& b)
      : id(b.id)
    {
      b.id = 0;
    }

    ~Buffer()
    {
      glDeleteBuffers(1, &id);
    }

    Buffer& operator =(Buffer&& b)
    {
      using std::swap;
      swap(id, b.id);
      return *this;
    }

    void swap(Buffer& b)
    {
      using std::swap;
      swap(id, b.id);
    }

    operator GLuint() const { return id; }
  };


  using std::swap;
  using std::move;
  using std::forward;
}

namespace std
{
  template <>
  inline void swap<GL::VertexArrayObject>(GL::VertexArrayObject& a, GL::VertexArrayObject& b)
  {
    a.swap(b);
  }

  template <>
  inline void swap<GL::Buffer>(GL::Buffer& a, GL::Buffer& b)
  {
    a.swap(b);
  }

}

#endif  // INCLUDED_GL_UTILS_BUFFER
