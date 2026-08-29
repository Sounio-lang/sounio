abi <abi/4.0>,
#include <tunables/global>

profile sounio-loom-kernel-peer-backend-discovery-v12 flags=(attach_disconnected,mediate_deleted) {
  /var/tmp/loom-kernel-peer-backend-discovery-v12-*/loom-peer-backend-discovery rix,
  /var/tmp/loom-kernel-peer-backend-discovery-v12-*/** rw,

  deny signal (receive) set=(term) peer=unconfined,
  deny ptrace (tracedby) peer=unconfined,
}
