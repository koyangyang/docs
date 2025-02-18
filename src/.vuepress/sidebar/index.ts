import { sidebar } from "vuepress-theme-hope";

import { openSourceProject } from "./open-source-tools.js";
import { computerBase } from "./computer-base.js";
import { algorithmLearn } from "./algorithm-learn.js";
import { dataScience } from "./data-science.js";

export default sidebar({
    // 应该把更精确的路径放置在前边
    "/开发工具/": openSourceProject,
    "/计算机基础/": computerBase,
    "/算法/": algorithmLearn,
    "/数据科学/": dataScience,
    // 必须放在最后面
    "/": [
        {
            text: "编程语言",
            icon: "book",
            collapsible: true,
            prefix: "language/",
            children: [
                {
                    text: "Go",
                    prefix: "golang/",
                    icon: "book",
                    children: [
                        {
                            text: "Go语言基础",
                            icon: "book",
                            collapsible: true,
                            link: "golangbase",
                        },
                        {
                            text: "重要知识点",
                            icon: "star",
                            collapsible: true,
                            children: [
                                "golangbase",
                            ],
                        },
                    ],
                },

            ],
        },
    ]
});
