# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2024 mlfoundations
#
# This code is based on https://github.com/mlfoundations/dclm

def is_factory(func):
    return hasattr(func, '_is_factory')


def factory_function(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper._is_factory = True
    return wrapper


def initialize_mapper(func, **kwargs):
    if is_factory(func):
        return func(**kwargs)
    return func
