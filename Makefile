all:
	cmake -B build -S .
	cmake --build build

clean:
	cmake --build build --target clean
