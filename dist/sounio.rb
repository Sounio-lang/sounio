# Homebrew formula for Sounio
# Install: brew install sounio/tap/sounio
# Or: brew tap sounio/tap && brew install sounio

class Sounio < Formula
  desc "L0 systems and scientific programming language for epistemic computing"
  homepage "https://github.com/sounio/sounio"
  license "MIT"
  version "0.97.0"

  on_macos do
    on_arm do
      url "https://github.com/sounio/sounio/releases/download/v#{version}/souc-aarch64-apple-darwin.tar.gz"
      sha256 "PLACEHOLDER_SHA256_ARM64_MACOS"
    end
    on_intel do
      url "https://github.com/sounio/sounio/releases/download/v#{version}/souc-x86_64-apple-darwin.tar.gz"
      sha256 "PLACEHOLDER_SHA256_X64_MACOS"
    end
  end

  on_linux do
    on_arm do
      url "https://github.com/sounio/sounio/releases/download/v#{version}/souc-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "PLACEHOLDER_SHA256_ARM64_LINUX"
    end
    on_intel do
      url "https://github.com/sounio/sounio/releases/download/v#{version}/souc-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "PLACEHOLDER_SHA256_X64_LINUX"
    end
  end

  def install
    bin.install "souc"
  end

  test do
    # Write a simple test program
    (testpath/"hello.sio").write <<~EOS
      fn main() -> i32 {
          0
      }
    EOS
    system "#{bin}/souc", "check", "hello.sio"
  end
end
