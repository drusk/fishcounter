# Copyright (C) 2013 David Rusk
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

"""
Provides various ways to prune spurious tracked objects.
"""


class Pruner(object):
    def _get_others(self, excluded_obj, others1, others2):
        all_others = list(others1)
        all_others.extend(others2)
        all_others.remove(excluded_obj)
        return all_others

    def prune_inactive(self, tracked_objects, current_frame_number, inactive_frames=1):

        def is_active(tracked_object):
            return (tracked_object.last_frame_tracked >
                    current_frame_number - inactive_frames)

        return filter(is_active, tracked_objects)

    def prune_subsumed(self, pruneable_objects, other_objects):
        sub_tracks = []
        for obj in pruneable_objects:
            for other_obj in self._get_others(obj, pruneable_objects, other_objects):
                if other_obj.contains(obj):
                    sub_tracks.append(obj)
                    break

        for obj in sub_tracks:
            print "Pruned subsumed"
            pruneable_objects.remove(obj)

        return pruneable_objects

    def prune_high_overlap(self, pruneable_objects, other_objects):
        obj_to_prune = []
        for obj in pruneable_objects:
            for other_obj in self._get_others(obj, pruneable_objects, other_objects):
                if other_obj.is_new():
                    continue

                if obj.bbox.area > other_obj.bbox.area:
                    # only prune the smaller objects
                    continue

                if obj.bbox_overlap_area(other_obj) > 0.5 * obj.bbox.area:
                    # This object is mostly overlapped with another
                    obj_to_prune.append(obj)
                    break # inner loop

        for obj in obj_to_prune:
            print "Pruned high overlap"
            pruneable_objects.remove(obj)

        return pruneable_objects

    def prune_super_objects(self, pruneable_objects, other_objects):
        obj_to_prune = set()
        for obj in pruneable_objects:
            for other_obj in other_objects:
                overlap = obj.bbox_overlap_area(other_obj)
                if overlap > 0.5 * other_obj.bbox.area:
                    obj_to_prune.add(obj)

        for obj in obj_to_prune:
            print "Pruned super track"
            pruneable_objects.remove(obj)

        return pruneable_objects

