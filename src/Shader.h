#ifndef SHADER_H
#define SHADER_H

#pragma once

#include <FileReader.h>
#include <Texture.h>

#include <VectorMath.h>

#include <GL3/gl3w.h>
#include <GL/utils/gl.h>
#include <GL/utils/gl_shader.h>

#include <string>
#include <map>
#include <vector>

class Shader
{
private:
	struct Subroutine
	{
		unsigned int index;
		std::string name;

	};

	Shader(const Shader&);
	Shader& operator =(Shader&);

	GL::VertexShader vertexShader;
	GL::FragmentShader fragmentShader;
	GL::Program program;

	std::map<unsigned int, std::vector<Subroutine> > subroutinesByShaderType;

public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	Shader(const std::string& vertexShaderFile, const std::string& fragmentShaderFile) : vertexShader(FileReader::read(vertexShaderFile).c_str()), fragmentShader(FileReader::read(fragmentShaderFile).c_str())
	{
		program.attachShader(vertexShader);
		program.attachShader(fragmentShader);
		program.link();
		GL_CHECK_ERROR();
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline void identifySubroutine(const std::string& name, unsigned int shaderType)
	{
		Subroutine subroutine;
		subroutine.name = name;
		subroutine.index = glGetSubroutineIndex(program, shaderType, name.c_str());
		GL_CHECK_ERROR();

		std::map<unsigned int, std::vector<Subroutine> >::iterator it = subroutinesByShaderType.find(shaderType);
		if (it != subroutinesByShaderType.end())
		{
			it->second.push_back(subroutine);
		}
		else
		{
			std::vector<Subroutine> subroutines;
			subroutines.push_back(subroutine);
			subroutinesByShaderType.insert(std::make_pair(shaderType, subroutines));
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline void setSubroutine(const std::string& name, unsigned int shaderType)
	{
		std::map<unsigned int, std::vector<Subroutine> >::iterator it = subroutinesByShaderType.find(shaderType);
		std::vector<Subroutine>& subroutines = it->second;
		for (unsigned int i = 0; i < subroutines.size(); i++)
		{
			Subroutine& subroutine = subroutines[i];
			if (subroutine.name == name)
			{
				glUniformSubroutinesuiv(shaderType, 1, &subroutine.index);
				return;
			}
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline void bind() const
	{
		glUseProgram(program);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline void unbind() const
	{
		glUseProgram(0);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline void bindAttribLocation(const std::string& name, unsigned int index)
	{
		glBindAttribLocation(program, index, name.c_str());
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline void setMat4(const std::string& name, const vml_mat4& uniform) const
	{
		glUniformMatrix4fv(glGetUniformLocation(program, name.c_str()), 1, GL_FALSE, &uniform[0][0]);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline void setVec2(const std::string& name, const vml_vec2& uniform) const
	{
		glUniform2fv(glGetUniformLocation(program, name.c_str()), 1, &uniform[0]);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline void setVec4(const std::string& name, const vml_vec4& uniform) const
	{
		glUniform4fv(glGetUniformLocation(program, name.c_str()), 1, &uniform[0]);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline void setFloat(const std::string& name, float uniform) const
	{
		glUniform1f(glGetUniformLocation(program, name.c_str()), uniform);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline void setTexture(const std::string& name, unsigned int textureId, unsigned int textureUnit) const
	{
		glActiveTexture(GL_TEXTURE0 + textureUnit);
		glBindTexture(GL_TEXTURE_2D, textureId);
		glUniform1i(glGetUniformLocation(program, name.c_str()), textureUnit);
	}

};

#endif