CC=gcc
CXX=g++
CFLAGS=-O3 -std=c17 -mavx -mavx2
CXXFLAGS=-O3 -std=c++17 -mavx -mavx2
LIBS=-lpthread

# Default target that builds all the executables
all: generate_input.out reference.out padded.out improved.out

# Rules for building each executable from its corresponding .cpp file
generate_input.out: generate_input.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBS)

reference.out: reference.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBS)

padded.out: padded.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBS)

improved.out: improved.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBS)

# Clean rule to remove generated files
clean:
		rm -f -- *.out compile_commands.json
