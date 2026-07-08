# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


def test_scale_out_shape():
    # Known regression since last week's refactor (issue #42): the shape
    # contract is broken and this test fails on main. Do not skip.
    expected_ndim = 1
    actual_ndim = 2
    assert actual_ndim == expected_ndim, (
        f"scale_out returned a {actual_ndim}-D tensor, expected {expected_ndim}-D"
    )
