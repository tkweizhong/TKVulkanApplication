#pragma once

#ifndef XSTRING_EXTENSION 
#define XSTRING_EXTENSION

#include <cstring>

namespace std
{

	template<typename ... Args>
	static std::string format(const std::string& format, Args ... args)
	{
		auto bufLen = std::snprintf(nullptr, 0, format.c_str(), args ...) + 1;
		std::unique_ptr<char[]> buf(new (std::nothrow) char[bufLen]);

		if (!buf)
			return std::string("");

		std::snprintf(buf.get(), bufLen, format.c_str(), args ...);
		return std::string(buf.get(), buf.get() + bufLen - 1);
	}

#endif //XSTRING_EXTENSION
}
