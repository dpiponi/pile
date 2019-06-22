SDK = xcrun -sdk macosx

all: explode.metallib explode

explode.metallib: explode.metal
	# Metal intermediate representation (.air)
	$(SDK) metal -O3 -c -Wall -Wextra -std=osx-metal2.0 -o /tmp/explode.air $^
	# Metal library (.metallib)
	$(SDK) metallib -o $@ /tmp/explode.air

explode: main.swift explode.swift
	$(SDK) swiftc -g -o $@ $^

clean:
	rm -f explode.metallib explode
