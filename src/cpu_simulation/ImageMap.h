#ifndef IMAGEMAP_H
#define IMAGEMAP_H

#include <FreeImage.h>
#include <glm/glm.hpp>

#include <string>

class ImageMap
{
public:
	ImageMap() : width(0), height(0), data(0), color1(0.0f, 0.0f, 0.0f, 1.0f), color2(1.0f, 1.0f, 1.0f, 1.0f) {}
	~ImageMap()
	{
		if (data != 0)
		{
			delete[] data;
		}
	}

	void import(const std::string& filePath, int desiredWidth, int desiredHeight)
	{
		if (data != 0)
		{
			delete[] data;
		}

		FREE_IMAGE_FORMAT format = FreeImage_GetFileType(filePath.c_str(), 0);
		FIBITMAP* bitmap = FreeImage_Load(format, filePath.c_str());
		FIBITMAP* image = FreeImage_ConvertTo32Bits(bitmap);
		int originalWidth = FreeImage_GetWidth(image);
		int originalHeigth = FreeImage_GetHeight(image);

		if (originalWidth != desiredWidth || originalHeigth != desiredHeight)
		{
			image = FreeImage_Rescale(image, desiredWidth, desiredHeight, FILTER_BILINEAR);
		}

		width = desiredWidth;
		height = desiredHeight;
		int size = width * height;
		data = new unsigned char[size];
		unsigned char* bgra = (unsigned char*)FreeImage_GetBits(image);

		for (int i = 0, j = 0; i < size; i++, j += 4)
		{
			// grayscale = (0.21 R + 0.71 G + 0.07 B)
			data[i] = (unsigned char)(0.21f * bgra[j + 2] + 0.71f * bgra[j + 1] + 0.07f * bgra[j]);
		}

		FreeImage_Unload(image);
	}

	bool castRay(const glm::vec3& origin, const glm::vec3& direction, unsigned int length, unsigned char threshold, glm::vec3& hit) const
	{
		for (unsigned int i = 0; i <= length; i++)
		{
			glm::vec3 point = origin + (direction * (float)i);

			if (sample(point) > threshold) 
			{
				hit = point;
				return false;
			}
		}

		return true;
	}

	bool castRay(const glm::vec3& origin, const glm::vec3& direction, unsigned int length, unsigned char threshold) const
	{
		glm::vec3 hit;
		return castRay(origin, direction, length, threshold, hit);
	}

	void scan(const glm::vec3& origin, const glm::vec3& direction, int minDistance, int maxDistance, unsigned char& greaterSample, int& distance) const
	{
		distance = maxDistance;
		greaterSample = 0;
		for (int i = minDistance; i <= maxDistance; i++)
		{
			glm::vec3 point = origin + (direction * (float)i);

			unsigned char currentSample = sample(point);
			if (currentSample > greaterSample) 
			{
				distance = i;
				greaterSample = currentSample;
			}
		}
	}

	unsigned char sample(const glm::vec3& point) const
	{
		glm::vec3 position;

		// clamp
		position.x = glm::max(0.0f, glm::min(point.x, (float)width));
		position.y = glm::max(0.0f, glm::min(point.y, (float)height));

		int i = ((int)position.y * width) + (int)position.x;

		return data[i];
	}

	inline int getWidth() const
	{
		return width;
	}

	inline int getHeight() const
	{
		return height;
	}

	inline const unsigned char* getData() const
	{
		return data;
	}

	inline unsigned char* getData()
	{
		return data;
	}

	inline glm::vec4 getColor1() const
	{
		return color1;
	}

	inline void setColor1(const glm::vec4& color)
	{
		color1 = color;
	}

	inline glm::vec4 getColor2() const
	{
		return color2;
	}

	inline void setColor2(const glm::vec4& color)
	{
		color2 = color;
	}

private:
	int width;
	int height;
	unsigned char* data;
	glm::vec4 color1;
	glm::vec4 color2;

};

#endif