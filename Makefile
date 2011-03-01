CC = g++
QTDIR=/usr/lib64/qt4
EIGEN=/afs/l2f.inesc-id.pt/home/ferreira/face-recognition/eigen

CLASSES = Database
PROGRAMS = eig3Train eig3Rec createDatabaseTester

SRCFILES = $(CLASSES:%=%.cpp) $(PROGRAMS:%=%.cpp)
OCLASSES = $(CLASSES:%=%.o)
OFILES = $(PROGRAMS:%=%.o) $(OCLASSES)

LIBEFJ = libefj.so

# optimize
CXXFLAGS = -I$(EIGEN) -I$(QTDIR)/include/QtCore/ -I$(QTDIR)/include/QtGui -DPIC -fPIC -m64 -pipe -DNDEBUG -DEIGEN_NO_DEBUG -O3 -msse2 -msse3 -mssse3 -msse4 -msse4.1 -msse4.2 -fmessage-length=0 -Wall -D_FORTIFY_SOURCE=2 -fstack-protector -funwind-tables -fasynchronous-unwind-tables -D_REENTRANT -fopenmp
# debug
#CXXFLAGS = -I$(EIGEN) -I$(QTDIR)/include/QtCore/ -I$(QTDIR)/include/QtGui -m64 -pipe -fmessage-length=0 -O2 -Wall -D_FORTIFY_SOURCE=2 -fstack-protector -funwind-tables -fasynchronous-unwind-tables -D_REENTRANT

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
	