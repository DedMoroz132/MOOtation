# NIST StRD, nonlinear regression

NIST Standard Reference Data (SRD 140); © U.S. Secretary of Commerce on behalf
of the United States of America.

The 27 files of NIST's Statistical Reference Datasets for nonlinear
regression, as NIST serves them (downloaded 2026-10-30 from
`https://itl.nist.gov/div898/strd/nls/data/LINKS/DATA/<name>.dat`; the page of
the datasets: https://itl.nist.gov/div898/strd/nls/nls_main.shtml). The bytes
are unchanged (CRLF line ends included; `.gitattributes` keeps them so);
`tests/test_lm_sets.cpp` checks each file's sha256 below before it reads it.
Each file carries both starts, the certified parameters and residual sum of
squares, and the data; `include/mootation/suite/nist.hpp` reads them and holds
the models.

| file | difficulty | parameters | observations | sha256 |
|---|---|---|---|---|
| `Misra1a.dat` | lower | 2 | 14 | `9d3194d537d9a6bbf92eb819504070923a1885d0191f40b6b3aaeedd130b98e2` |
| `Chwirut2.dat` | lower | 3 | 54 | `c37e5522a34f9a57e18d7d10f8736202eabc0c33b8e7ad0010e618db04b57411` |
| `Chwirut1.dat` | lower | 3 | 214 | `d9a055dfe5af71a8754c00f073ef00f8fed2e3fd1c6fd20cea8fd62d7cc3ed84` |
| `Lanczos3.dat` | lower | 6 | 24 | `bc998624541c49fedee08346ee2c1d495741c4ea1dd15c052d7c743ab3cf0efe` |
| `Gauss1.dat` | lower | 8 | 250 | `20d0b85ffc73681ce9bdd69c566e53969a65e0e2192c3de16ad4a1dcfdca6c27` |
| `Gauss2.dat` | lower | 8 | 250 | `24fe86801c35c4f86140cced68894e31d73b6ed5738cf698474144f94bae3640` |
| `DanWood.dat` | lower | 2 | 6 | `5eae83de8b34e34c631a5738d640183b0628913e728437ae77f066e832186e24` |
| `Misra1b.dat` | lower | 2 | 14 | `dd27e310f787dcbc2addc50b9f8e5e3b82155f732926898b1858e4d08eead6a8` |
| `Kirby2.dat` | average | 5 | 151 | `cbd288dd9d8586ec97ca408277d1a27ef41b19b543c158d594af189de80ae61c` |
| `Hahn1.dat` | average | 7 | 236 | `7e885af48e18a222033a66cb22b491f0f5bc761b652142c0d9b28132daf63956` |
| `Nelson.dat` | average | 3 | 128 | `82fe77ef1a79a4441e7af5083680fc5046f9eb29b33bc2f2b76c478ecbe976d7` |
| `MGH17.dat` | average | 5 | 33 | `75edb5522ee639794ae1f0d0f4a1e3638e963c6854888b853e1f45cea3c2f4d7` |
| `Lanczos1.dat` | average | 6 | 24 | `dc7a419a4d7e7c0891e378dcde953a3e5dd273618b64b2cee8301e51577b428a` |
| `Lanczos2.dat` | average | 6 | 24 | `be1ac732e725ef75298f082bcdedab9e2ee9395ad77d8fad2ef37c95e0421572` |
| `Gauss3.dat` | average | 8 | 250 | `24842a9e795db59c2c6cbc4c11047f90faedad014f8146faa6b5555259958d6d` |
| `Misra1c.dat` | average | 2 | 14 | `012f75aa0ff112afabdff73880e324ab28e2dc267b678537611c403b959dc941` |
| `Misra1d.dat` | average | 2 | 14 | `4dfa39a381a5ffea3e5d9306af1bda9d53bc57226a8e6f0c19af6b9702580481` |
| `Roszman1.dat` | average | 4 | 25 | `fb526231ae6b02929e3de54e072e259b3d87424089abd4b1015ffb946b1e0ab0` |
| `ENSO.dat` | average | 9 | 168 | `0c644f45b0e586cb1bb0613bad68f63c704efadacd086e768468d33d83f3e5d9` |
| `MGH09.dat` | higher | 4 | 11 | `6419da83fd9ab34615769413dc2a6d35b11372f512a9401628a1565067c51d10` |
| `Thurber.dat` | higher | 7 | 37 | `806b8d00a5365eff907561db796fec3af8a26a5f353297d4311fd46ed5b0146f` |
| `BoxBOD.dat` | higher | 2 | 6 | `fc44fbed04e3be16df6498e722d4c8aedfd0af210110bc396d9f99acecd77e21` |
| `Rat42.dat` | higher | 3 | 9 | `fc97dd1f94fbcb381495065a053e279f23588a9ceb243276f2a2c36c8225985b` |
| `MGH10.dat` | higher | 3 | 16 | `a0eb140580fdd6dcf6fba2cbbd81afa598c5226f1e6d702d1313c72a7a6e6fb4` |
| `Eckerle4.dat` | higher | 3 | 35 | `8ec2133b29a0a7cecbfccd51f7c61ee432f73c460bf5ce7106d672642e57a61e` |
| `Rat43.dat` | higher | 4 | 15 | `36f08908d0f09b1d075e3701ca567701535c58ac0d795ff68fb3c04e172c2c5d` |
| `Bennett5.dat` | higher | 3 | 154 | `b34d160605b959c301d13aac6627b693743d32ad98a6c4f830a3f728cdff0f52` |
