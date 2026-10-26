// SPDX-License-Identifier: Apache-2.0
// NumPy's .npz read in C++ (include/mootation/io/npz.hpp) and a reference cache
// entry with its hash (campaign/references.hpp), on files NumPy and zlib wrote
// (tests/data/npz, tools/refs/gen_npz_samples.py): inflate of a stored, a
// fixed and a dynamic block; a compressed and a stored archive; float64, int64,
// Fortran order and a unicode string; reference_hash as Python computes it.

#include <cmath>
#include <cstdio>
#include <string>

#include <mootation/campaign/references.hpp>
#include <mootation/io/npz.hpp>

namespace npz = mootation::npz;
namespace cg = mootation::campaign;

static int failures = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
            ++failures;                                                   \
        }                                                                 \
    } while (0)

static const std::string DATA = std::string(MOOTATION_TEST_DATA) + "/npz/";

int main()
{
    // one text deflated in each block type: its length and CRC-32 (zlib's)
    for (const char* name : {"stored", "fixed", "dyn"}) {
        const auto z = npz::read_file(DATA + name + ".deflate");
        const auto text = npz::inflate(z.data(), z.size());
        CHECK(text.size() == 15866);
        CHECK(mootation::store::crc32_zlib(text.data(), text.size()) == 424854747u);
    }
    // a damaged stream is refused, not read wrong
    {
        auto z = npz::read_file(DATA + "dyn.deflate");
        z.resize(z.size() / 2);
        bool refused = false;
        try {
            npz::inflate(z.data(), z.size());
        } catch (const npz::Error&) {
            refused = true;
        }
        CHECK(refused);
    }

    // numpy.savez_compressed of a cache entry
    const auto a = npz::load(DATA + "entry.npz");
    CHECK(a.size() == 5 && a.count("meta") && a.count("ref") && !a.count("ref3"));
    CHECK(a.at("meta").text() == "{\"problem\": \"T\\u00e9st\", \"n_ref\": 40, \"info3\": null}");
    const auto& ref = a.at("ref");
    CHECK(ref.descr == "<f8" && ref.shape.size() == 2 && ref.shape[0] == 40 && ref.shape[1] == 3);
    const auto r = ref.doubles();
    CHECK(r[3] == 0.02564102564102564 && r[4] == 0.9743589743589743 && r[5] == 0.16012815380508713);
    const cg::ReferenceEntry e = cg::load_entry(DATA + "entry.npz");
    CHECK(e.meta.find("problem") && e.meta.find("problem")->s == "T\xc3\xa9st");
    CHECK(e.ideal && e.nadir && e.ref && !e.ref3 && e.pset && e.pset->shape[0] == 8);
    CHECK(cg::reference_hash(e.ideal, e.nadir, e.ref, e.ref3, e.pset) == "a10d17240165dcf5");
    CHECK(cg::shape_text({}) == "()" && cg::shape_text({3}) == "(3,)" && cg::shape_text({1000, 3}) == "(1000, 3)");

    // numpy.savez: stored members, int64 and a Fortran-ordered array read in C order
    const auto s = npz::load(DATA + "stored.npz");
    CHECK(s.at("ints").descr == "<i8" && s.at("ints").doubles() == std::vector<double>({0, 1, 2, 3, 4}));
    CHECK(s.at("F").fortran_order && s.at("F").doubles() == std::vector<double>({0, 1, 2, 3, 4, 5}));

    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("npz: NumPy's archives and zlib's streams\n");
    return 0;
}
