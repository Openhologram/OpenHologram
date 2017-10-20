#include "graphics/ComputeShader.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
namespace graphics {


	ComputeShader::ComputeShader()
		: shader_src_(0),
		shader_handle_(0),
		program_id_(0)
	{
	}

	ComputeShader::~ComputeShader()
	{
		if (shader_src_) {
			free(shader_src_);
			glDeleteShader(shader_handle_);
		}
		glDeleteProgram(program_id_);
	}

	GLhandleARB ComputeShader::LoadShader(const char* source)
	{
		GLhandleARB handle;

		// shader Compilation variable
		GLint result;				// Compilation code result
		GLint errorLoglength;
		char* errorLogText;
		GLsizei actualErrorLogLength;

		handle = glCreateShader(GL_COMPUTE_SHADER);
		if (!handle)
		{
			//We have failed creating the vertex shader object.
			LOG("Failed creating compute shader object.");
		}

		glShaderSource(
			handle, //The handle to our shader
			1, //The number of files.
			&source, //An array of const char * data, which represents the source code of theshaders
			NULL);

		glCompileShader(handle);

		//Compilation checking.
		glGetObjectParameterivARB(handle, GL_OBJECT_COMPILE_STATUS_ARB, &result);

		// If an error was detected.
		if (!result)
		{
			//We failed to compile.
			LOG("Shader failed compilation.\n");

			//Attempt to get the length of our error log.
			glGetObjectParameterivARB(handle, GL_OBJECT_INFO_LOG_LENGTH_ARB, &errorLoglength);

			//Create a buffer to read compilation error message
			errorLogText = (char*)malloc(sizeof(char) * errorLoglength);

			//Used to get the final length of the log.
			glGetInfoLogARB(handle, errorLoglength, &actualErrorLogLength, errorLogText);

			// Display errors.
			LOG("%s\n", errorLogText);

			// Free the buffer malloced earlier
			free(errorLogText);
		}

		return handle;
	}


	void
	ComputeShader::BeginShader()
	{

		glUseProgram(program_id_);
	}


	void
	ComputeShader::EndShader()
	{
		glUseProgram(0);
	}



	void
		ComputeShader::Initialize()
	{


		program_id_ = glCreateProgram();

		if (shader_src_)
			shader_handle_ = LoadShader(shader_src_);


		glAttachShader(program_id_, shader_handle_);

		glLinkProgram(program_id_);
	}

	void ComputeShader::SetShaderSource(const char* source)
	{
		if (!source) return;
		int length = strlen(source);
		shader_src_ = (char*)malloc(length + 10);
		strcpy(shader_src_, source);
	}

	void ComputeShader::SetShaderSourceWithFile(const char* fname)
	{
		std::string src = "";
		std::ifstream fin;

		fin.open(fname);

		if (!fin.good())
		{
			LOG("Can not open shader source file ");
			return;
		}

		while (fin.good())
		{
			char buf[1024];
			fin.getline(buf, 1024);
			src += buf;
			src += "\n";
		}

		src += "\0";

		fin.close();

		shader_src_ = (char*)malloc(src.length() * sizeof(char) + 10);
		memset((void*)shader_src_, 0, src.length() * sizeof(char) + 10);
		strcpy(shader_src_, src.c_str());

	}
};