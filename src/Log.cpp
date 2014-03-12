#include <Log.h>

std::map<std::string, Logger*> Log::s_loggers;
std::string FileLogger::s_lineEnd = "\n";