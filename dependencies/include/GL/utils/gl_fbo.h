#ifndef INCLUDED_GL_UTILS_FBO
#define INCLUDED_GL_UTILS_FBO

#pragma once

#include <utility>

#include "gl.h"
#include "gl_texture.h"

namespace GL
{
  class RenderBuffer
  {
  private:
    RenderBuffer(const RenderBuffer&);
	
    RenderBuffer& operator =(const RenderBuffer&);
    GLuint id;
	
    static GLuint create()
    {
      GLuint id;
      glGenRenderbuffers(1, &id);
      GL_CHECK_ERROR();
      if (id == 0)
        throw error("couldn't create renderbuffer");
      return id;
    }
  
  public:
    RenderBuffer() : id(create())
    {
    }

    RenderBuffer(GLuint id)
      : id(id)
    {
    }

    RenderBuffer(RenderBuffer&& tex)
      : id(tex.id)
    {
      tex.id = 0;
    }

    RenderBuffer(GLsizei width, GLsizei height, GLenum format) : id(create())
    {
      glBindRenderbuffer(GL_RENDERBUFFER, *this);
      glRenderbufferStorage(GL_RENDERBUFFER, format, width, height);
      GL_CHECK_ERROR();
    }
  
    RenderBuffer(GLsizei width, GLsizei height, GLenum format, GLsizei samples) : id(create())
    {
      glBindRenderbuffer(GL_RENDERBUFFER, *this);
      glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, format, width, height);
      GL_CHECK_ERROR();
    }

    ~RenderBuffer()
    {
      glDeleteRenderbuffers(1, &id);
    }

    RenderBuffer& operator =(RenderBuffer&& tex)
    {
      using std::swap;
      swap(id, tex.id);
      return *this;
    }

    void swap(RenderBuffer& b)
    {
      using std::swap;
      swap(id, b.id);
    }

    operator GLuint() const { return id; }
  };

  class FrameBuffer
  {
  private:
    FrameBuffer(const FrameBuffer&);
  
    FrameBuffer& operator =(const FrameBuffer&);
    GLuint id;
	
    static GLuint create()
    {
      GLuint id;
      glGenFramebuffers(1, &id);
      GL_CHECK_ERROR();
      if (id == 0)
        throw error("couldn't create framebuffer");
      return id;
    }
	
  public:
    FrameBuffer()
      : id(create())
    {
    }

    FrameBuffer(GLuint id)
      : id(id)
    {
    }

    FrameBuffer(FrameBuffer&& fbo)
      : id(fbo.id)
    {
      fbo.id = 0;
    }

    ~FrameBuffer()
    {
      glDeleteFramebuffers(1, &id);
    }

    FrameBuffer& operator =(FrameBuffer&& fbo)
    {
      using std::swap;
      swap(id, fbo.id);
      return *this;
    }

    void swap(FrameBuffer& b)
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
  inline void swap<GL::RenderBuffer>(GL::RenderBuffer& a, GL::RenderBuffer& b)
  {
    a.swap(b);
  }

  template <>
  inline void swap<GL::FrameBuffer>(GL::FrameBuffer& a, GL::FrameBuffer& b)
  {
    a.swap(b);
  }
}

#endif  // INCLUDED_GL_UTILS_FBO
