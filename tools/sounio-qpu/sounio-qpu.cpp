// Transport beside souc. Posts a QIR text file and writes the shot list
// tape_read already understands. The listen command is the stand-in for
// the device on the other end of the socket. It is not a quantum processor.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot read " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string shot_list(long shots) {
    if (shots < 2 || (shots % 2) != 0) {
        throw std::runtime_error("shots must be a positive even count");
    }
    const long half = shots / 2;
    std::string s;
    s.reserve(static_cast<size_t>(shots) * 3);
    for (long i = 0; i < half; ++i) s += "00\n";
    for (long i = 0; i < half; ++i) s += "11\n";
    return s;
}

int bind_listen(int port) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) throw std::runtime_error("socket");
    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        throw std::runtime_error("bind");
    }
    if (listen(fd, 1) < 0) throw std::runtime_error("listen");
    return fd;
}

std::string recv_all(int fd) {
    std::string data;
    char buf[4096];
    while (true) {
        const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n == 0) break;
        if (n < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error("recv");
        }
        data.append(buf, static_cast<size_t>(n));
        const auto hdr = data.find("\r\n\r\n");
        if (hdr == std::string::npos) continue;
        const auto cl = data.find("Content-Length:");
        if (cl == std::string::npos || cl > hdr) break;
        long len = std::stol(data.substr(cl + 15));
        if (static_cast<long>(data.size() - (hdr + 4)) >= len) break;
    }
    return data;
}

long header_long(const std::string& req, const char* key) {
    const auto pos = req.find(key);
    if (pos == std::string::npos) return -1;
    return std::stol(req.substr(pos + std::strlen(key)));
}

size_t message_end(const std::string& data) {
    const auto hdr = data.find("\r\n\r\n");
    if (hdr == std::string::npos) return std::string::npos;
    const auto cl = data.find("Content-Length:");
    if (cl == std::string::npos || cl > hdr) return hdr + 4;
    const long len = std::stol(data.substr(cl + 15));
    if (len < 0) return std::string::npos;
    const size_t need = hdr + 4 + static_cast<size_t>(len);
    if (data.size() >= need) return need;
    return std::string::npos;
}

std::string recv_one(int fd, std::string& pending) {
    std::string data = std::move(pending);
    pending.clear();
    while (true) {
        const size_t end = message_end(data);
        if (end != std::string::npos) {
            pending = data.substr(end);
            return data.substr(0, end);
        }
        char buf[4096];
        const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n == 0) return data;
        if (n < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error("recv");
        }
        data.append(buf, static_cast<size_t>(n));
    }
}

void send_all(int fd, const std::string& bytes) {
    size_t off = 0;
    while (off < bytes.size()) {
        const ssize_t n = ::send(fd, bytes.data() + off, bytes.size() - off, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error("send");
        }
        off += static_cast<size_t>(n);
    }
}

const char* connection_token(const std::string& req) {
    if (req.find("Connection: close") != std::string::npos) return "close";
    return "keep-alive";
}

std::string health_response(const std::string& req) {
    std::ostringstream out;
    out << "HTTP/1.1 200 OK\r\nConnection: " << connection_token(req)
        << "\r\nContent-Type: text/plain\r\nContent-Length: 3\r\n\r\nok\n";
    return out.str();
}

std::string shots_response(const std::string& req) {
    const long shots = header_long(req, "X-Shots:");
    const auto body_at = req.find("\r\n\r\n");
    if (shots < 2 || (shots % 2) != 0 || body_at == std::string::npos || req.size() < body_at + 5) {
        return "HTTP/1.1 400 Bad Request\r\nConnection: close\r\nContent-Length: 0\r\n\r\n";
    }
    const std::string body = shot_list(shots);
    std::ostringstream out;
    out << "HTTP/1.1 200 OK\r\nConnection: " << connection_token(req)
        << "\r\nContent-Type: text/plain\r\nContent-Length: " << body.size()
        << "\r\n\r\n" << body;
    return out.str();
}

void listen_once(int port) {
    const int srv = bind_listen(port);
    std::cout << "sounio-qpu listen " << port
              << " transport stand-in, not a device\n" << std::flush;
    sockaddr_in client{};
    socklen_t clen = sizeof(client);
    const int fd = accept(srv, reinterpret_cast<sockaddr*>(&client), &clen);
    if (fd < 0) throw std::runtime_error("accept");
    std::string pending;
    for (int n = 0; n < 8; ++n) {
        const std::string req = recv_one(fd, pending);
        if (req.find("\r\n\r\n") == std::string::npos) break;
        std::string resp;
        bool bad = false;
        if (req.compare(0, 11, "GET /health") == 0) {
            resp = health_response(req);
        } else if (req.compare(0, 11, "POST /shots") == 0) {
            resp = shots_response(req);
            bad = resp.find(" 400 ") != std::string::npos;
        } else {
            resp = "HTTP/1.1 400 Bad Request\r\nConnection: close\r\nContent-Length: 0\r\n\r\n";
            bad = true;
        }
        send_all(fd, resp);
        if (bad || req.find("Connection: close") != std::string::npos) break;
    }
    ::close(fd);
    ::close(srv);
}

int connect_to(const std::string& host, int port) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) throw std::runtime_error("socket");
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        throw std::runtime_error("bad host");
    }
    if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        throw std::runtime_error("connect");
    }
    return fd;
}

std::string response_body(const std::string& raw) {
    const auto hdr = raw.find("\r\n\r\n");
    if (hdr == std::string::npos) throw std::runtime_error("no http body");
    if (raw.find("200") == std::string::npos) throw std::runtime_error("device refused");
    const auto cl = raw.find("Content-Length:");
    if (cl == std::string::npos || cl > hdr) throw std::runtime_error("no content-length");
    const long len = std::stol(raw.substr(cl + 15));
    if (len < 0) throw std::runtime_error("bad content-length");
    return raw.substr(hdr + 4, static_cast<size_t>(len));
}

void submit(const std::string& qir_path, long shots, const std::string& host, int port,
            const std::string& out_path) {
    const std::string qir = read_file(qir_path);
    std::ostringstream req;
    req << "POST /shots HTTP/1.1\r\nHost: " << host << "\r\nX-Shots: " << shots
        << "\r\nContent-Type: text/plain\r\nContent-Length: " << qir.size()
        << "\r\nConnection: close\r\n\r\n" << qir;
    const int fd = connect_to(host, port);
    const std::string bytes = req.str();
    size_t sent = 0;
    while (sent < bytes.size()) {
        const ssize_t n = ::send(fd, bytes.data() + sent, bytes.size() - sent, 0);
        if (n < 0) throw std::runtime_error("send");
        sent += static_cast<size_t>(n);
    }
    const std::string body = response_body(recv_all(fd));
    ::close(fd);
    std::ofstream out(out_path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + out_path);
    out << body;
    std::cout << "sounio-qpu wrote " << out_path << " shots " << shots << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "listen") {
            listen_once(std::stoi(argv[2]));
            return 0;
        }
        if (argc == 12 && std::string(argv[1]) == "submit") {
            std::string qir, host, out;
            long shots = 0;
            int port = 0;
            for (int i = 2; i < argc; i += 2) {
                const std::string k = argv[i];
                const std::string v = argv[i + 1];
                if (k == "--qir") qir = v;
                else if (k == "--shots") shots = std::stol(v);
                else if (k == "--host") host = v;
                else if (k == "--port") port = std::stoi(v);
                else if (k == "--out") out = v;
                else throw std::runtime_error("unknown arg");
            }
            submit(qir, shots, host, port, out);
            return 0;
        }
    } catch (const std::exception& ex) {
        std::cerr << "sounio-qpu: " << ex.what() << "\n";
        return 1;
    }
    std::cerr << "usage: sounio-qpu listen PORT\n"
              << "       sounio-qpu submit --qir FILE --shots N --host H --port P --out FILE\n";
    return 2;
}
