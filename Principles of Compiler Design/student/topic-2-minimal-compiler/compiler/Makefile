# ============================================================================
# mini compiler — build rules
# ----------------------------------------------------------------------------
#   make            build ./minicompiler
#   make test       compile and run every program in tests/ under SPIM
#   make clean      remove everything that `make` produced
#
# The two generated files are worth knowing about:
#   lex.yy.c        written by flex from scanner.l
#   parser.tab.c/.h written by bison from parser.y
# Never edit those two — edit the .l and .y files and rebuild.
# ============================================================================
CC     = gcc
LEX    = flex
YACC   = bison
CFLAGS = -g -Wall

TARGET = minicompiler
OBJS   = lex.yy.o parser.tab.o main.o ast.o symtab.o semantic.o codegen.o tac.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

# The scanner needs the token numbers that bison assigns, so parser.tab.h
# must exist before flex output is compiled.
lex.yy.c: scanner.l parser.tab.h
	$(LEX) scanner.l

parser.tab.c parser.tab.h: parser.y
	$(YACC) -d parser.y

lex.yy.o: lex.yy.c
	$(CC) $(CFLAGS) -c lex.yy.c

parser.tab.o: parser.tab.c
	$(CC) $(CFLAGS) -c parser.tab.c

main.o:     main.c ast.h semantic.h codegen.h tac.h trace.h
	$(CC) $(CFLAGS) -c main.c
ast.o:      ast.c ast.h
	$(CC) $(CFLAGS) -c ast.c
symtab.o:   symtab.c symtab.h trace.h
	$(CC) $(CFLAGS) -c symtab.c
semantic.o: semantic.c semantic.h ast.h symtab.h trace.h
	$(CC) $(CFLAGS) -c semantic.c
codegen.o:  codegen.c codegen.h ast.h symtab.h tac.h trace.h
	$(CC) $(CFLAGS) -c codegen.c
tac.o:      tac.c tac.h ast.h trace.h
	$(CC) $(CFLAGS) -c tac.c

# ----------------------------------------------------------------------------
# Compile every tests/*.cm file and run the result in SPIM.  Each test file
# states its expected output in a header comment; compare and you have a
# regression suite.
# ----------------------------------------------------------------------------
test: $(TARGET)
	@for f in tests/*.cm; do \
	   echo "── $$f ─────────────────────────────────────────"; \
	   ./$(TARGET) $$f $${f%.cm}.s -q > /dev/null || { echo "  COMPILE FAILED"; continue; }; \
	   spim -file $${f%.cm}.s 2>/dev/null | grep -v '^Loaded'; \
	done

clean:
	rm -f $(TARGET) $(OBJS) lex.yy.c parser.tab.c parser.tab.h parser.output
	rm -f tests/*.s tests/*.tac *.s *.tac

.PHONY: all clean test
