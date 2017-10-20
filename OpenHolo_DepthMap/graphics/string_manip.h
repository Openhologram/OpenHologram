#ifndef __STRING_MANIP_H
#define __STRING_MANIP_H

#include <string>
#include <vector>
namespace graphics
{
	std::basic_string<wchar_t> ToLower(const std::basic_string<wchar_t>& source);
	std::basic_string<wchar_t> ToUpper(const std::basic_string<wchar_t>& source);
	std::basic_string<wchar_t> ReplaceAll(const std::basic_string<wchar_t> &str, const std::basic_string<wchar_t>& pattern, const std::basic_string<wchar_t>& replace);
	// c:\a\b.txt -> c:/a/
	// c:\a\b\c\d -> c:/a/b/c 를 반환한다.
	std::basic_string<wchar_t> GetDirectoryPath(const std::basic_string<wchar_t>& path);
	// 경로의 마지막 단계에 있는 디렉토리 명이나 파일명을 반환한다. 파일명은 확장자 포함를 포함한다.
	std::basic_string<wchar_t> GetPathLeafName(const std::basic_string<wchar_t>& path);

	void SeparateNameAndExt(const std::basic_string<wchar_t>& path, std::basic_string<wchar_t>& file_name, std::basic_string<wchar_t>& ext);

	// 경로는 반드시 디렉토리여야 한다.
	std::basic_string<wchar_t> AbsolutePathToRelativePath(const std::basic_string<wchar_t>& absolutePath, const std::basic_string<wchar_t>& targetPath);

	void get_tokens(std::string input, std::vector<std::string>& tokens, const char* delim);

	void separate_string(std::string input, std::string& out1, std::string& out2, const char* delim);

	void stripoff_blank(std::string& input, std::string& out);

	std::string to_utf8(const std::wstring& input);
	std::wstring to_utf16(const std::string& input);
}

#endif