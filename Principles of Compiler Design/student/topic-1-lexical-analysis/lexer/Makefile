# ============================================================================
# Topic 1 · Lexical analyzer
#   make          build ./lexer
#   make test     run it over every file in tests/
#   make clean    remove generated files
# ============================================================================
CC     = gcc
LEX    = flex
CFLAGS = -g -Wall

TARGET = lexer

all: $(TARGET)

$(TARGET): lex.yy.o main.o
	$(CC) $(CFLAGS) -o $(TARGET) lex.yy.o main.o

lex.yy.c: scanner.l tokens.h
	$(LEX) scanner.l

lex.yy.o: lex.yy.c tokens.h
	$(CC) $(CFLAGS) -c lex.yy.c

main.o: main.c tokens.h
	$(CC) $(CFLAGS) -c main.c

test: $(TARGET)
	@for f in tests/*.cm; do \
	   echo "══ $$f ═══════════════════════════════════════════"; \
	   ./$(TARGET) $$f -c; \
	done

clean:
	rm -f $(TARGET) *.o lex.yy.c

.PHONY: all test clean
