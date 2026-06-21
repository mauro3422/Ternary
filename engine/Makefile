CC = gcc
CFLAGS = -msse4.2 -O2 -Wall -Wextra -Wno-unused-parameter
LDFLAGS = -lm

.PHONY: all clean

all: tern size

tern: main.c
	$(CC) $(CFLAGS) main.c -o tern $(LDFLAGS)

# Verificar que el binario sea pequeño
size: tern
	@echo "Tamaño del binario:"
	@ls -lh tern
	@echo "Dependencias:"
	@ldd tern 2>/dev/null || echo "  (estático, sin dependencias)"

# Cross-compile para N4020 (Linux x86_64)
tern-linux: main.c
	x86_64-linux-gnu-gcc -static -msse4.2 -O2 main.c -o tern-linux -lm
	@echo "Tamaño (Linux):"
	@ls -lh tern-linux

clean:
	rm -f tern tern-linux
