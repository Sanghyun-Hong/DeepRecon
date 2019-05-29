# Ingredients
FILES=$(wildcard *.c)
TARGETS=$(FILES:.c=)
OBJS=${FILES:.c=.o}

# Location variable
PYTHON_LIB=$(shell python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")

# Flags
CFLAGS=-g -std=gnu99 -I../Mastik/src
LDFLAGS=-L../Mastik/src/ -L${PYTHON_LIB}/tensorflow/python -g
LDLIBS=-lmastik -ldwarf -lelf -lbfd -lm ${PYTHON_LIB}/tensorflow/python/_pywrap_tensorflow_internal.so

# Make-commands
all: ${TARGETS}
${TARGETS}: %: %.o 
	${CC} ${LDFLAGS} -o $@ $@.o ${LDLIBS}

clean:
	rm -f ${TARGETS} ${OBJS}
	rm -f *.csv *.txt

distclean: clean
	rm Makefile
