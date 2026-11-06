The weekly Madaros prebuilt refresh workflow now writes its own
`changelog.d/` fragment alongside the rebuilt `bin/madaros-linux-x86_64`, so the
pull request it opens satisfies the changelog fragment gate and can reach the
merge queue. Before this, every refresh PR failed Impact and cascaded into
Contracts and CI Decision, so a refreshed compiler could never ship.
