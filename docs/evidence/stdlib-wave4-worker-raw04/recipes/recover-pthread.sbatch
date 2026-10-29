#!/bin/bash
set -euo pipefail
R=/scratch/sounio-stdlib-wave3-20261027/root-integrated04
D=$R/pthread-package-v3
mkdir "$D";cd "$D"
trap 'r=$?;echo "$r" >STATUS;exit "$r"' EXIT
url=https://archive.ubuntu.com/ubuntu/pool/main/g/glibc/libc6-dev_2.39-0ubuntu8.7_amd64.deb
printf '%s\n' "$url" >ORIGIN_URL
curl --fail --location --proto '=https' --connect-timeout 15 --max-time 120 -D DOWNLOAD_HEADERS "$url" -o libc6-dev_2.39-0ubuntu8.7_amd64.deb
sha256sum libc6-dev_2.39-0ubuntu8.7_amd64.deb >PACKAGE.sha256
dpkg-deb -f libc6-dev_2.39-0ubuntu8.7_amd64.deb Package Version Architecture >PACKAGE_METADATA
test "$(dpkg-deb -f libc6-dev_2.39-0ubuntu8.7_amd64.deb Package)" = libc6-dev
test "$(dpkg-deb -f libc6-dev_2.39-0ubuntu8.7_amd64.deb Version)" = 2.39-0ubuntu8.7
test "$(dpkg-deb -f libc6-dev_2.39-0ubuntu8.7_amd64.deb Architecture)" = amd64
dpkg-deb -x libc6-dev_2.39-0ubuntu8.7_amd64.deb root
sha256sum root/usr/lib/x86_64-linux-gnu/libpthread.a >PTHREAD.sha256
sha256sum -c PACKAGE.sha256 >PACKAGE.final.verify
sha256sum -c PTHREAD.sha256 >PTHREAD.final.verify
echo OFFICIAL_LIBC6_DEV_PTHREAD_EXTRACTED
