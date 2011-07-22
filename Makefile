# $Id: Makefile,v 1.9 2011/07/22 14:44:23 david Exp $
#
# Copyright (C) 2008-2011 INESC ID Lisboa.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
#
# $Log: Makefile,v $
# Revision 1.9  2011/07/22 14:44:23  david
# Minor cleanup.
#
#

CC = g++
#32bit
#QTDIR=/usr/lib/qt4
#64bit
QTDIR=/usr/lib64/qt4
EIGEN=/afs/l2f.inesc-id.pt/home/ferreira/face-recognition/eigen

CLASSES = Database Database_io Database_debug
PROGRAMS = 

SRCFILES = $(CLASSES:%=%.cpp) $(PROGRAMS:%=%.cpp)
OCLASSES = $(CLASSES:%=%.o)
OFILES = $(PROGRAMS:%=%.o) $(OCLASSES)

LIBEFJ = libefj.so

# 32bit
#BASE_CXXFLAGS = -I$(EIGEN) -I$(QTDIR)/include/QtCore/ -I$(QTDIR)/include/QtGui -DPIC -fPIC -m32 -pipe 
# 64bit
BASE_CXXFLAGS = -I$(EIGEN) -I$(QTDIR)/include/QtCore/ -I$(QTDIR)/include/QtGui -DPIC -fPIC -m64 -pipe 
# 32bit vanilla
#CXXFLAGS = $(BASE_CXXFLAGS) -DNDEBUG -DEIGEN_NO_DEBUG -O3 -fmessage-length=0 -Wall -D_FORTIFY_SOURCE=2 -fstack-protector -funwind-tables -fasynchronous-unwind-tables -D_REENTRANT -fopenmp

# optimize
#CXXFLAGS = $(BASE_CXXFLAGS) -DDEBUG -DNDEBUG -DEIGEN_NO_DEBUG -O3 -msse2 -msse3 -mssse3 -msse4 -msse4.1 -msse4.2 -fmessage-length=0 -Wall -D_FORTIFY_SOURCE=2 -fstack-protector -funwind-tables -fasynchronous-unwind-tables -D_REENTRANT -fopenmp
# debug
CXXFLAGS = $(BASE_CXXFLAGS) -ggdb -Wall -DDEBUG -D_FORTIFY_SOURCE=2 -funwind-tables -fasynchronous-unwind-tables -D_REENTRANT

LDFLAGS = -L. -lefj -lboost_filesystem -lQtGui -lgomp

all: $(LIBEFJ) $(PROGRAMS)

$(LIBEFJ): $(OCLASSES)
	$(CXX) -shared -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

%: %.o
	$(CXX) -o $@ $< $(LDFLAGS)

clean: 
	$(RM) $(PROGRAMS) $(OFILES) $(LIBEFJ)
	
depend:
	$(CXX) $(CXXFLAGS) -MM $(SRCFILES) > .makedeps

-include .makedeps
	
