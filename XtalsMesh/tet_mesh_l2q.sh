#! /bin/bash
#
g++ -c -Wall /work/tet_mesh_l2q.cpp
if [ $? -ne 0 ]; then
  echo "Compile error."
  exit
fi
#
g++ /work/tet_mesh_l2q.o -lm
if [ $? -ne 0 ]; then
  echo "Load error."
  exit
fi
rm tet_mesh_l2q.o
#
chmod ugo+x a.out
mv a.out /work/tet_mesh_l2q
#
echo "Normal end of execution."
