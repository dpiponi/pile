SDK = xcrun -sdk macosx

all: pile.metallib pile

pile.metallib: pile.metal
	# Metal intermediate representation (.air)
	$(SDK) metal -O3 -c -Wall -Wextra -std=osx-metal2.0 -o /tmp/pile.air $^
	# Metal library (.metallib)
	$(SDK) metallib -o $@ /tmp/pile.air

pile: main.swift pile.swift
	$(SDK) swiftc -g -o $@ $^

clean:
	rm -f pile.metallib pile
