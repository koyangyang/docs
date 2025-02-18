import { arraySidebar } from "vuepress-theme-hope";

export const algorithmLearn = arraySidebar([
    {
        text: "算法",
        icon: "book",
        prefix: "Leetcode/",
        collapsible: true,
        children: [
            {
                text: "Leetcode面试热门150题",
                // icon: "book",
                collapsible: true,
                link: "Leetcode面试热门150题",
            },
        ],
    },
]);
