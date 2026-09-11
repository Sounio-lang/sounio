#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

namespace pireus::material {

class Sha256 {
  public:
    Sha256() = default;

    void update(const std::uint8_t* data, std::size_t size) {
        bit_count_ += static_cast<std::uint64_t>(size) * 8U;
        while (size > 0) {
            const std::size_t room = block_.size() - block_size_;
            const std::size_t take = size < room ? size : room;
            for (std::size_t i = 0; i < take; ++i) {
                block_[block_size_ + i] = data[i];
            }
            block_size_ += take;
            data += take;
            size -= take;
            if (block_size_ == block_.size()) {
                compress(block_.data());
                block_size_ = 0;
            }
        }
    }

    void update_i64_be(std::int64_t value) {
        std::array<std::uint8_t, 8> bytes{};
        const std::uint64_t bits = static_cast<std::uint64_t>(value);
        for (std::size_t i = 0; i < bytes.size(); ++i) {
            bytes[i] = static_cast<std::uint8_t>(bits >> (56U - i * 8U));
        }
        update(bytes.data(), bytes.size());
    }

    std::array<std::uint32_t, 8> finish() const {
        Sha256 copy = *this;
        const std::uint64_t message_bits = copy.bit_count_;
        const std::uint8_t marker = 0x80U;
        copy.update(&marker, 1);
        const std::uint8_t zero = 0;
        while (copy.block_size_ != 56) {
            copy.update(&zero, 1);
        }
        std::array<std::uint8_t, 8> length{};
        for (std::size_t i = 0; i < length.size(); ++i) {
            length[i] = static_cast<std::uint8_t>(
                message_bits >> (56U - i * 8U));
        }
        copy.update(length.data(), length.size());
        return copy.state_;
    }

  private:
    static constexpr std::array<std::uint32_t, 64> kRound = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
        0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
        0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
        0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
    };

    static constexpr std::uint32_t rotate_right(std::uint32_t value,
                                                 unsigned count) {
        return (value >> count) | (value << (32U - count));
    }

    static std::uint32_t load_be(const std::uint8_t* bytes) {
        return (static_cast<std::uint32_t>(bytes[0]) << 24U) |
            (static_cast<std::uint32_t>(bytes[1]) << 16U) |
            (static_cast<std::uint32_t>(bytes[2]) << 8U) |
            static_cast<std::uint32_t>(bytes[3]);
    }

    void compress(const std::uint8_t* bytes) {
        std::array<std::uint32_t, 64> words{};
        for (std::size_t i = 0; i < 16; ++i) {
            words[i] = load_be(bytes + i * 4);
        }
        for (std::size_t i = 16; i < words.size(); ++i) {
            const std::uint32_t s0 = rotate_right(words[i - 15], 7) ^
                rotate_right(words[i - 15], 18) ^ (words[i - 15] >> 3U);
            const std::uint32_t s1 = rotate_right(words[i - 2], 17) ^
                rotate_right(words[i - 2], 19) ^ (words[i - 2] >> 10U);
            words[i] = words[i - 16] + s0 + words[i - 7] + s1;
        }

        std::uint32_t a = state_[0];
        std::uint32_t b = state_[1];
        std::uint32_t c = state_[2];
        std::uint32_t d = state_[3];
        std::uint32_t e = state_[4];
        std::uint32_t f = state_[5];
        std::uint32_t g = state_[6];
        std::uint32_t h = state_[7];
        for (std::size_t i = 0; i < words.size(); ++i) {
            const std::uint32_t sum1 = rotate_right(e, 6) ^
                rotate_right(e, 11) ^ rotate_right(e, 25);
            const std::uint32_t choice = (e & f) ^ ((~e) & g);
            const std::uint32_t temporary1 =
                h + sum1 + choice + kRound[i] + words[i];
            const std::uint32_t sum0 = rotate_right(a, 2) ^
                rotate_right(a, 13) ^ rotate_right(a, 22);
            const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temporary2 = sum0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temporary1;
            d = c;
            c = b;
            b = a;
            a = temporary1 + temporary2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_ = {
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
    };
    std::array<std::uint8_t, 64> block_{};
    std::uint64_t bit_count_ = 0;
    std::size_t block_size_ = 0;
};

inline std::string digest_words(const std::array<std::uint32_t, 8>& words) {
    std::ostringstream output;
    for (std::size_t i = 0; i < words.size(); ++i) {
        if (i != 0) {
            output << ':';
        }
        output << static_cast<std::uint64_t>(words[i]);
    }
    return output.str();
}

inline std::string digest_hex(const std::array<std::uint32_t, 8>& words) {
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (const std::uint32_t word : words) {
        output << std::setw(8) << word;
    }
    return output.str();
}

}  // namespace pireus::material
