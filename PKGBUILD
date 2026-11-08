pkgname=mass-reencode
pkgver=1.1.0
pkgrel=1
pkgdesc="Find the least space-efficient videos in a directory and re-encode them with ffmpeg (completely vibe-coded)"
arch=('any')
url="https://github.com/NotVibeCodedSlop/mass-reencode"
license=('custom')
depends=('ffmpeg' 'python')
source=('optimize_worst.py')
sha256sums=('SKIP')

package() {
    install -Dm755 "$srcdir/optimize_worst.py" "$pkgdir/usr/bin/mass-reencode"
}
