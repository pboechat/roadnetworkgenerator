#ifndef FONT_H
#define FONT_H

#include <stdio.h>
#include <exception>
#include <string>

//////////////////////////////////////////////////////////////////////////
class GLFontBase
{
public:
	GLFontBase() : initialized(0), buffers(0), vaos(0)
	{
	}

	virtual ~GLFontBase()
	{
		freeResources();
	}

protected:
	void createImpl(const std::string& filePath, unsigned int textureId, bool pixelPerfect = 0)
	{
		font.characters = 0;
		freeResources();
		FILE* file;

		if ((file = fopen(filePath.c_str(), "rb")) == 0)
		{
			throw std::exception("invalid font file");
		}

		fread(&font, sizeof(Font), 1, file);
		font.texture = textureId;
		numChars = font.end - font.start + 1;
		font.characters = new Character[numChars];
		fread(font.characters, sizeof(Character), numChars, file);
		int textureSize = font.textureWidth * font.textureHeight * 2;
		char* textureData = new char[textureSize];
		fread(textureData, sizeof(char), textureSize, file);
		glBindTexture(GL_TEXTURE_2D, font.texture);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		if (pixelPerfect)
		{
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		}

		else
		{
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		}

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RG16, font.textureWidth, font.textureHeight, 0, GL_RG, GL_UNSIGNED_BYTE, (void*)textureData);
		delete []textureData;
		fclose(file);
		vaos = new unsigned int[numChars];
		glGenVertexArrays(numChars, vaos);
		numBuffers = numChars * 2;
		buffers = new unsigned int[numBuffers];
		glGenBuffers(numBuffers, buffers);

		for (unsigned int i = 0, j = 0; i < numChars; i++, j += 2)
		{
			Character* character = &font.characters[i];
			float vertices[] = { character->dx, character->dy, 0.0f, 1.0f,
								 0.0f, character->dy, 0.0f, 1.0f,
								 0.0f, 0.0f, 0.0f, 1.0f,
								 character->dx, 0.0f, 0.0f, 1.0f
							   };
			glBindBuffer(GL_ARRAY_BUFFER, buffers[j]);
			glBufferData(GL_ARRAY_BUFFER, 16 * sizeof(float), (void*)vertices, GL_STATIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			float uvs[] = { character->tx2, character->ty1,
							character->tx1, character->ty1,
							character->tx1, character->ty2,
							character->tx2, character->ty2
						  };
			glBindBuffer(GL_ARRAY_BUFFER, buffers[j + 1]);
			glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(float), (void*)uvs, GL_STATIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			glBindVertexArray(vaos[i]);
			glBindBuffer(GL_ARRAY_BUFFER, buffers[j]);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			glBindBuffer(GL_ARRAY_BUFFER, buffers[j + 1]);
			glEnableVertexAttribArray(1);
			glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			glBindVertexArray(0);
		}

		initialized = 1;
	}

	//////////////////////////////////////////////////////////////////////////
	typedef struct
	{
		union
		{
			float dx;
			int width;
		};
		union
		{
			float dy;
			int height;
		};

		float tx1, ty1;
		float tx2, ty2;
	} Character;

	//////////////////////////////////////////////////////////////////////////
	typedef struct
	{
		int texture;
		int textureWidth, textureHeight;
		int start, end;
		Character* characters;
	} Font;

	Font font;
	unsigned int* vaos;
	unsigned int* buffers;
	unsigned int numChars;
	unsigned int numBuffers;
	bool initialized;

private:
	void freeResources ()
	{
		if (font.characters != 0)
		{
			delete[] font.characters;
		}

		if (buffers != 0)
		{
			glDeleteBuffers(numBuffers, buffers);
			delete[] buffers;
		}

		if (vaos != 0)
		{
			glDeleteVertexArrays(numChars, vaos);
			delete[] vaos;
		}

		initialized = 0;
	}
};

//////////////////////////////////////////////////////////////////////////
class GLFont : public GLFontBase
{
public:
	//////////////////////////////////////////////////////////////////////////
	struct FontSetter
	{
		virtual void operator()(float x, float y, float z, float scale, const float* color, unsigned int textureId) = 0;

	};

	GLFont() {}

	void create(const std::string& filePath, unsigned int textureId)
	{
		GLFontBase::createImpl(filePath, textureId, 0);
	}

	void draw(std::string text, float x, float y, float z, float scale, const float* color, FontSetter& setter)
	{
		static unsigned int indices[] = { 0, 1, 2, 0, 2, 3 };

		if (!initialized)
		{
			throw std::exception("font uninitialized");
		}

		int length = text.length();

		for (int i = 0; i < length; i++)
		{
			int j = (int)text[i] - font.start;
			Character* character = &font.characters[j];
			setter(x, y, z, scale, color, font.texture);
			glBindVertexArray(vaos[j]);
			glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, (void*)indices);
			glBindVertexArray(0);
			x += (character->dx * scale);
		}
	}
};

#endif


