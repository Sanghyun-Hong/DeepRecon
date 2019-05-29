#!/bin/bash

MASTIK_DIR=Mastik
PYTHON_LIB=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")

# run...
echo "--------------------------------------"
echo " : Extract the Tensorflow information"
echo "--------------------------------------"
objcopy --only-keep-debug "$PYTHON_LIB/tensorflow/python/_pywrap_tensorflow_internal.so" "objdumps/_pywrap_tf_internal.so.debug" && sleep 5s


echo "--------------------------------------"
echo " : Mastik confs. before compilation"
echo "--------------------------------------"
# Note: do not run 'make clean' in the Mastik dir - this will remove the objdump file
cd $MASTIK_DIR; ./configure
cd src
sed -i '2s/.*/LIBSRCS=vlist.c l3.c timestats.c l1.c l1i.c fr.c util.c pda.c symbol.c ff.c symbol_bfd.c ..\/..\/objdumps\/_pywrap_tf_internal.so.debug/' Makefile
cd ../demo
PYHLIB_STR=$(echo "$PYTHON_LIB" | sed 's./.\\/.g')
sed -i '5s/.*/LDFLAGS=-L..\/src\/ -L'$PYHLIB_STR'\/tensorflow\/python -g/' Makefile
sed -i '6s/.*/LDLIBS=-lmastik -ldwarf -lelf -lbfd '$PYHLIB_STR'\/tensorflow\/python\/_pywrap_tensorflow_internal.so/' Makefile
cd ..

echo "--------------------------------------"
echo " : Mastik confs. before compilation"
echo "--------------------------------------"
make
# done.
