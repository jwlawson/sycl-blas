
%module tuner
%{
#include "get_time_for_config.hpp"
%}


%include get_time_for_config.hpp
%template(get_time_for) get_time_for_config<float>;
