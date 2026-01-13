#include <stdio.h>

int searchInsert(int nums[], int n, int target) {
    int left = 0, right = n - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (nums[mid] == target) {
            return mid;   // target found
        } else if (nums[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    // target not found, left is the insertion index
    return left;
}

int main() {
    int nums[] = {1, 3, 5, 6};
    int n = sizeof(nums) / sizeof(nums[0]);

    int target = 5;
    printf("Index: %d\n", searchInsert(nums, n, target));

    target = 2;
    printf("Index: %d\n", searchInsert(nums, n, target));

    target = 7;
    printf("Index: %d\n", searchInsert(nums, n, target));

    return 0;
}
