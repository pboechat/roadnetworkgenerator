#ifndef FILEREADER_H_
#define FILEREADER_H_

#include <string>
#include <exception>
#include <stdio.h>

#include <string>

class FileReader
{
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	static std::string read(const std::string& fileName)
	{
		char* buffer = 0;

		if (!fileName.empty())
		{
			FILE* file = fopen(fileName.c_str(), "rt");

			if (file == NULL)
			{
				throw std::exception("File not found");
			}

			fseek(file, 0, SEEK_END);
			int bytesRead = ftell(file);
			rewind(file);

			if (bytesRead > 0)
			{
				buffer = (char*)malloc(sizeof(char) * (bytesRead + 1));
				bytesRead = fread(buffer, sizeof(char), bytesRead, file);
				buffer[bytesRead] = '\0';
			}

			fclose(file);
		}

		std::string fileContent = "";

		if (buffer != 0)
		{
			fileContent = buffer;
			free(buffer);
		}

		return fileContent;
	}

private:
	FileReader() {}
	~FileReader() {}

};

#endif