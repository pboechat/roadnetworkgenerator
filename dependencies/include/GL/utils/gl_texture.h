#ifndef INCLUDED_GL_UTILS_TEXTURE
#define INCLUDED_GL_UTILS_TEXTURE

#pragma once

#include <utility>

#include "gl.h"

namespace
{
  inline size_t typeSize(GLenum type)
  {
    switch (type)
    {
      case GL_BYTE:
      case GL_UNSIGNED_BYTE:
    	return 1;
      case GL_SHORT:
      case GL_UNSIGNED_SHORT:
    	return 2;
      case GL_INT:
      case GL_UNSIGNED_INT:
      case GL_FLOAT:
    	return 4;
      case GL_DOUBLE:
    	return 8;
    }
    return -1;
  }
  
  inline int channelCount(GLenum format)
  {
    switch (format)
    {
      case GL_RED:
    	return 1;
      case GL_RG:
    	return 2;
      case GL_RGB:
      case GL_BGR:
    	return 3;
      case GL_RGBA:
      case GL_BGRA:
    	return 4;
    }
    return -1;
  }
  
  inline size_t pixelSize(GLenum format, GLenum type)
  {
    return typeSize(type) * channelCount(format);
  }
}

namespace GL
{
  class Texture
  {
  private:
    Texture(const Texture&);
    Texture& operator =(const Texture&);
    GLuint id;
    static GLuint create()
    {
      GLuint id;
      glGenTextures(1, &id);
      GL_CHECK_ERROR();
      if (id == 0)
        throw error("couldn't create texture");
      return id;
    }
  protected:
    Texture()
      : id(create())
    {
    }

    Texture(GLuint id)
      : id(id)
    {
    }

    Texture(Texture&& tex)
      : id(tex.id)
    {
      tex.id = 0;
    }

    ~Texture()
    {
      glDeleteTextures(1, &id);
    }

    Texture& operator =(Texture&& tex)
    {
      using std::swap;
      swap(id, tex.id);
      return *this;
    }

    void swap(Texture& t)
    {
      using std::swap;
      swap(id, t.id);
    }

  public:
    operator GLuint() const { return id; }
  };

  class Texture1D : public Texture
  {
  private:
    Texture1D(const Texture1D&);
    Texture1D& operator =(const Texture1D&);
    using Texture::swap;
  public:
    Texture1D()
	{
	}
	
    Texture1D(GLuint id)
      : Texture(id)
    {
    }
	
    Texture1D(GLsizei width, GLint levels, GLenum format)//, GLenum data_format, GLenum data_type, const GLvoid* data)
    {
      glBindTexture(GL_TEXTURE_1D, *this);
      glTexStorage1D(GL_TEXTURE_1D, levels, format, width);
    }
  
    Texture1D(Texture1D&& tex)
      : Texture(std::forward<Texture>(tex))
    {
    }

    Texture1D& operator =(Texture1D&& tex)
    {
      Texture::operator =(std::forward<Texture>(tex));
      return *this;
    }

    void swap(Texture1D& t)
    {
      swap(static_cast<Texture&>(t));
    }
  };

  class Texture2D : public Texture
  {
  private:
#ifdef _WIN32
    Texture2D(const Texture2D&);
    Texture2D& operator =(const Texture2D&);
#else
    Texture2D(const Texture2D&) = delete;
    Texture2D& operator =(const Texture2D&) = delete;
#endif
    using Texture::swap;
  public:
    Texture2D()
    {
    }
	
    Texture2D(GLuint id)
      : Texture(id)
    {
    }
	
    Texture2D(GLsizei width, GLsizei height, GLint levels, GLenum format)
    {
      glBindTexture(GL_TEXTURE_2D, *this);
      glTexStorage2D(GL_TEXTURE_2D, levels, format, width, height);
      GL_CHECK_ERROR();
    }
  
    Texture2D(GLsizei width, GLsizei height, GLint levels, GLenum format, GLenum data_format, GLenum data_type, const GLvoid* data)
    {
      glBindTexture(GL_TEXTURE_2D, *this);
      glTexStorage2D(GL_TEXTURE_2D, levels, format, width, height);
      GL_CHECK_ERROR();
    
      if (data != nullptr)
      {
        const char* ptr = static_cast<const char*>(data);
        int w = width;
        int h = height;
        const size_t pixel_size = pixelSize(data_format, data_type);
        for (int level = 0; level < levels; ++level)
        {
          glTexSubImage2D(GL_TEXTURE_2D, level, 0, 0, w, h, data_format, data_type, ptr);
          GL_CHECK_ERROR();
          ptr += w * h * pixel_size;
          w >>= 1;
          h >>= 1;
        }
      }
    }
  
    Texture2D(Texture2D&& tex)
      : Texture(std::forward<Texture>(tex))
    {
    }

    Texture2D& operator =(Texture2D&& tex)
    {
      Texture::operator =(std::forward<Texture>(tex));
      return *this;
    }

    void swap(Texture2D& t)
    {
      swap(static_cast<Texture&>(t));
    }
  };

  class Texture2DArray : public Texture
  {
  private:
    Texture2DArray(const Texture2DArray&);
    Texture2DArray& operator =(const Texture2DArray&);

    using Texture::swap;
  public:
    Texture2DArray()
    {
    }
	
    Texture2DArray(GLuint id)
      : Texture(id)
    {
    }
	
    Texture2DArray(GLsizei width, GLsizei height, GLint slices, GLint levels, GLenum format)
    {
      glBindTexture(GL_TEXTURE_2D_ARRAY, *this);
      glTexStorage3D(GL_TEXTURE_2D_ARRAY, levels, format, width, height, slices);
    }
  
    Texture2DArray(Texture2DArray&& tex)
      : Texture(std::forward<Texture>(tex))
    {
    }

    Texture2DArray& operator =(Texture2DArray&& tex)
    {
      Texture::operator =(std::forward<Texture>(tex));
      return *this;
    }

    void swap(Texture2DArray& t)
    {
      swap(static_cast<Texture&>(t));
    }
  };

  class Texture2DMS : public Texture
  {
  private:
    Texture2DMS(const Texture2DMS&);
    Texture2DMS& operator =(const Texture2DMS&);
    using Texture::swap;
  public:
    Texture2DMS()
	{
	}
	
    Texture2DMS(GLuint id)
	  : Texture(id)
	{
	}
	
    Texture2DMS(GLsizei width, GLsizei height, GLenum format, GLsizei samples);
    Texture2DMS(Texture2DMS&& tex)
      : Texture(std::forward<Texture>(tex))
    {
    }

    Texture2DMS& operator =(Texture2DMS&& tex)
    {
      Texture::operator =(std::forward<Texture>(tex));
      return *this;
    }

    void swap(Texture2DMS& t)
    {
      swap(static_cast<Texture&>(t));
    }
  };


  class Texture3D : public Texture
  {
  private:
    Texture3D(const Texture3D&);
    Texture3D& operator =(const Texture3D&);
    using Texture::swap;
  public:
    Texture3D()
	{
	}
	
    Texture3D(GLuint id) :
	  Texture(id)
	{
	}
	
    Texture3D(GLsizei width, GLsizei height, GLsizei depth, GLint levels, GLenum format)//, GLenum data_format, GLenum data_type, const GLvoid* data)
    {
      glBindTexture(GL_TEXTURE_3D, *this);
      glTexStorage3D(GL_TEXTURE_3D, levels, format, width, height, depth);
    }
    Texture3D(Texture3D&& tex)
      : Texture(std::forward<Texture>(tex))
    {
    }

    Texture3D& operator =(Texture3D&& tex)
    {
      Texture::operator =(std::forward<Texture>(tex));
      return *this;
    }

    void swap(Texture3D& t)
    {
      swap(static_cast<Texture&>(t));
    }
  };

  class TextureCube : public Texture
  {
  private:
    TextureCube(const TextureCube&);
    TextureCube& operator =(const TextureCube&);
    using Texture::swap;
  public:
    TextureCube()
	{
	}
	
    TextureCube(GLuint id)
	  : Texture(id)
	{
	}
	
    TextureCube(GLsizei width, GLint levels, GLenum format)//, GLenum data_format, GLenum data_type, const GLvoid* data)
    {
      glBindTexture(GL_TEXTURE_CUBE_MAP, *this);
      glTexStorage2D(GL_TEXTURE_CUBE_MAP, levels, format, width, width);
    }
  
    TextureCube(TextureCube&& tex)
      : Texture(std::forward<Texture>(tex))
    {
    }

    TextureCube& operator =(TextureCube&& tex)
    {
      Texture::operator =(std::forward<Texture>(tex));
      return *this;
    }

    void swap(TextureCube& t)
    {
      swap(static_cast<Texture&>(t));
    }
  };

  class SamplerState
  {
  private:
    SamplerState(const Texture&);
    SamplerState& operator =(const SamplerState&);

    GLuint id;
    static GLuint create()
    {
      GLuint id;
      glGenSamplers(1, &id);
      GL_CHECK_ERROR();
      if (id == 0)
        throw error("couldn't create sampler object");
      return id;
    }
  public:
    SamplerState()
      : id(create())
    {
    }
    ~SamplerState()
    {
      glDeleteSamplers(1, &id);
    }

    void swap(SamplerState& s)
    {
      using std::swap;
      swap(id, s.id);
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
  inline void swap<GL::Texture1D>(GL::Texture1D& a, GL::Texture1D& b)
  {
    a.swap(b);
  }

  template <>
  inline void swap<GL::Texture2D>(GL::Texture2D& a, GL::Texture2D& b)
  {
    a.swap(b);
  }

  template <>
  inline void swap<GL::Texture2DMS>(GL::Texture2DMS& a, GL::Texture2DMS& b)
  {
    a.swap(b);
  }

  template <>
  inline void swap<GL::Texture3D>(GL::Texture3D& a, GL::Texture3D& b)
  {
    a.swap(b);
  }

  template <>
  inline void swap<GL::TextureCube>(GL::TextureCube& a, GL::TextureCube& b)
  {
    a.swap(b);
  }

  template <>
  inline void swap<GL::SamplerState>(GL::SamplerState& a, GL::SamplerState& b)
  {
    a.swap(b);
  }
}

#endif  // INCLUDED_GL_UTILS_TEXTURE
