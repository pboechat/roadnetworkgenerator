#ifndef TEXTURE_H
#define TEXTURE_H

#include <GL3/gl3w.h>
#include <FreeImage.h>

#include <string>
#include <exception>

class Texture
{
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	Texture(unsigned int width, unsigned int height, unsigned int format = GL_RGB, unsigned int internalFormat = GL_RGB32F, unsigned int type = GL_FLOAT, unsigned int filter = GL_LINEAR, unsigned int wrap = GL_CLAMP_TO_EDGE, void* data = 0) :
		id(0),
		width(width),
		height(height),
		format(format),
		internalFormat(internalFormat),
		type(type),
		filter(filter),
		wrap(wrap),
		data(data)
	{
		glGenTextures(1, &id);
		glBindTexture(GL_TEXTURE_2D, id);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap);
		glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, data);
		glGenerateMipmap(GL_TEXTURE_2D);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	~Texture()
	{
		glDeleteTextures(1, &id);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	static Texture* import(const std::string& fileName)
	{
		FREE_IMAGE_FORMAT format = FreeImage_GetFileType(fileName.c_str(), 0);
		FIBITMAP* bitmap = FreeImage_Load(format, fileName.c_str());
		FIBITMAP* image = FreeImage_ConvertTo32Bits(bitmap);
		int width = FreeImage_GetWidth(image);
		int height = FreeImage_GetHeight(image);
		// FreeImage loads in BGRA format, so swap some bytes to use RGBA format in OpenGL
		int size = 4 * width * height;
		unsigned char* data = new unsigned char[size];
		char* pixels = (char*)FreeImage_GetBits(image);

		for (int j = 0; j < size; j += 4)
		{
			data[j + 0] = pixels[j + 2];
			data[j + 1] = pixels[j + 1];
			data[j + 2] = pixels[j + 0];
			data[j + 3] = pixels[j + 3];
		}

		Texture* texture = new Texture(width, height, GL_RGBA, GL_RGBA32F, GL_UNSIGNED_BYTE, GL_LINEAR, GL_REPEAT, (void*)data);
		FreeImage_Unload(image);
		return texture;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	operator unsigned int()
	{
		return id;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline unsigned int getHeight() const
	{
		return height;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline unsigned int getWidth() const
	{
		return width;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline unsigned int getFormat() const
	{
		return format;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline unsigned int getInternalFormat() const
	{
		return internalFormat;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline unsigned int getType() const
	{
		return type;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline unsigned int getFilterMode() const
	{
		return filter;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline unsigned int getWrapMode() const
	{
		return wrap;
	}

private:
	static bool initialized;
	unsigned int width;
	unsigned int height;
	unsigned int format;
	unsigned int internalFormat;
	unsigned int type;
	unsigned int filter;
	unsigned int wrap;
	unsigned int id;
	void* data;

};

#endif