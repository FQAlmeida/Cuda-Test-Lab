all:
	cmake -B build -S .
	cmake --build build --verbose

clean:
	cmake --build build --target clean
