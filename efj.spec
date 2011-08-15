Name: libefj0
Version: 0.0
Release: 1

Summary: A library for face identification using eigenfaces
License: GPL
Group: Development/Libraries/C and C++
Url: http://robots.l2f.inesc-id.pt/

Source: efj-%{version}.tar.bz2
Prefix: %_prefix
BuildRoot: %{_tmppath}/efj-%{version}-build
BuildRequires: gcc-c++ libeigen3-devel boost-devel libqt4-devel

%description
A library for face identification using eigenfaces.

%package -n efj-devel
License:        GPL
Group:          Development/Libraries/C and C++
Summary:        The development files for libefj
BuildRequires:  gcc-c++ libeigen3-devel boost-devel libqt4-devel
Requires:       %{name} = %{version}
Requires:	gcc-c++ libeigen3-devel boost-devel libqt4-devel

%description -n efj-devel
A library for face identification using eigenfaces.

%prep
%setup -n efj-%{version}

%build
make LIBDIR=%{_libdir} INCLUDEDIR=%{_includedir} DATADIR=%{_datadir}

%install
mkdir -p $RPM_BUILD_ROOT%{_libdir}
cp libefj.a  $RPM_BUILD_ROOT%{_libdir}/libefj.a
cp libefj.so $RPM_BUILD_ROOT%{_libdir}/libefj.so.0.0
(cd $RPM_BUILD_ROOT%{_libdir}; ln -s libefj.so.0.0 libefj.so.0)
(cd $RPM_BUILD_ROOT%{_libdir}; ln -s libefj.so.0.0 libefj.so)
mkdir -p $RPM_BUILD_ROOT%{_includedir}/efj
cp Database.h $RPM_BUILD_ROOT%{_includedir}/efj
cp misc.h $RPM_BUILD_ROOT%{_includedir}/efj

# files
cd $RPM_BUILD_ROOT
find .%{_includedir} -print | sed 's,^\.,\%attr(-\,root\,root) ,'  > $RPM_BUILD_DIR/files-devel.list
echo .%{_libdir}/libefj.a   | sed 's,^\.,\%attr(-\,root\,root) ,' >> $RPM_BUILD_DIR/files-devel.list
echo .%{_libdir}/libefj.so  | sed 's,^\.,\%attr(-\,root\,root) ,' >> $RPM_BUILD_DIR/files-devel.list
echo .%{_libdir}/libefj.so.0.0 | sed 's,^\.,\%attr(-\,root\,root) ,'  > $RPM_BUILD_DIR/files.list
echo .%{_libdir}/libefj.so.0   | sed 's,^\.,\%attr(-\,root\,root) ,' >> $RPM_BUILD_DIR/files.list

%clean
rm -rf $RPM_BUILD_ROOT
rm -rf $RPM_BUILD_DIR/%{name}-%{version}
rm $RPM_BUILD_DIR/files.list
rm $RPM_BUILD_DIR/files-devel.list

%post   -p /sbin/ldconfig

%postun -p /sbin/ldconfig

%files -f ../files.list

%files -n efj-devel -f ../files-devel.list

%changelog
* Sat Aug 13 2011 david@inesc-id.pt
- efj on opensuse 11.4 with gcc 4.5

