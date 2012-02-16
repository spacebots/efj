# $Id: Makefile,v 1.12 2012/02/16 17:21:26 david Exp $
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
# Revision 1.12  2012/02/16 17:21:26  david
# Correcção de alguns bugs (David + Jaime).
#
# Revision 1.11  2012/02/12 02:05:23  ferreira
# Added CSUFaceIDEvalSystem compatible output
#
# Revision 1.10  2011/08/15 16:36:14  david
# Updated project files to be more compatible with building installation
# packages.
#
# Revision 1.9  2011/07/22 14:44:23  david
# Minor cleanup.
#
#

CC = c++

# default values (openSUSE 11.4, 64 bits)
LIBDIR=/usr/lib64
INCLUDEDIR=/usr/include

QTDIR=$(LIBDIR)/qt4
EIGEN=$(INCLUDEDIR)/eigen3

CLASSES = Database Database_io Database_debug Database_csuOutput

SRCFILES = $(CLASSES:%=%.cpp)
OCLASSES = $(CLASSES:%=%.o)
OFILES = $(OCLASSES)

LIBEFJ_SO = libefj.so
LIBEFJ_A = libefj.a

BASE_CXXFLAGS = -I. -I$(QTDIR)/include/QtCore/ -I$(QTDIR)/include/QtGui -I$(EIGEN) -DPIC -fPIC -pipe 
# optimize
#CXXFLAGS = $(BASE_CXXFLAGS) -DDEBUG -DNDEBUG -DEIGEN_NO_DEBUG -O3 -msse2 -msse3 -mssse3 -msse4 -msse4.1 -msse4.2 -fmessage-length=0 -Wall -D_FORTIFY_SOURCE=2 -fstack-protector -funwind-tables -fasynchronous-unwind-tables -D_REENTRANT -fopenmp
# debug
CXXFLAGS = $(BASE_CXXFLAGS) -ggdb -Wall -DDEBUG -D_FORTIFY_SOURCE=2 -funwind-tables -fasynchronous-unwind-tables -D_REENTRANT -lQtCore -lQtGui -lgomp -lboost_system

all: link $(LIBEFJ_SO) $(LIBEFJ_A)

link:
	-ln -s . efj

$(LIBEFJ_SO): $(OCLASSES)
	$(CXX) -shared -o $@ $^

$(LIBEFJ_A): $(OCLASSES)
	$(AR) crv $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

clean: 
	$(RM) $(PROGRAMS) $(OFILES) $(LIBEFJ_SO) $(LIBEFJ_A)
	
depend:
	$(CXX) $(CXXFLAGS) -MM $(SRCFILES) > .makedeps

-include .makedeps
	
