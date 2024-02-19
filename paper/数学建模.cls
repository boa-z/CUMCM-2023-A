% \NeedsTeXFormat{LaTeX2e}%指定了需要的LATEX格式和版本
% \ProvidesClass{math}%指定了文档类的名称为math，并提供了文档类的版本信息
% [2023/02/08 v0.2 Standard LaTeX Template for math Report]
% \LoadClass[UTF8,a4paper]{ctexart}%加载了一个名为ctexart的LaTeX文档类，采用UTF-8编码，a4paper选型设置纸张尺寸为A4
% \RequirePackage{graphicx}%加载了graphicx宏包，用于在文档中插入图形
% \RequirePackage{geometry}%用于设置页面布局和边距
% \RequirePackage{titlesec}%用于自定义章节标题的样式
% \RequirePackage{fancyhdr}%用于自定义页眉和页脚的样式
% \RequirePackage[most,minted]{tcolorbox}%tcolorbox用于创建带有颜色框和边框的文本框，minted是一个代码高亮宏包
% \RequirePackage{minted}%高亮代码
% \RequirePackage{amsmath}%提供一些数学符号和环境的增强功能
% \RequirePackage{unicode-math}%用于处理Unicode编码的数学符号
% \RequirePackage{multirow}%用于在表格中合并行
% \RequirePackage{array}%提供了对表格和数组的增强支持
% \RequirePackage{float}%提供了更好的浮动体控制
% \RequirePackage{indentfirst}%使得首段缩进
% \RequirePackage{zhnumber}%提供中文数字的格式化功能，比如将阿拉伯数字转换为中文数字
% \RequirePackage{booktabs}%提供了创建漂亮的表格的功能，包括不同粗细的水平线
% \RequirePackage{todonotes}%用于在文档中添加待办事项的注释
% \usepackage{fontspec}
% \usepackage{setspace}
% \usepackage{longtable}
% \usepackage{csvsimple}
% \usepackage{comment}
% \usepackage{graphicx}
% \usepackage{array}
% \usepackage{makecell}
% \usepackage{svg}
% \usepackage{subfigure}
% \usepackage{array}
% \geometry{a4paper,top=2.5cm,bottom=2.5cm,left=2.5cm,right=2.5cm}

% \unimathsetup{math-style=ISO, bold-style=ISO}


% \RequirePackage{kvdefinekeys}
% \RequirePackage{kvsetkeys}
% \RequirePackage{kvoptions}


% % 行间距
% \linespread{1.25}

% % 字体汉化设置
% \xeCJKsetup{EmboldenFactor=2.5} % 伪粗体等级
% % 字体设置
% \setmainfont{Times New Roman} 
% \renewcommand{\songti}{\CJKfontspec[Path=fonts/,FallBack=simsunb.ttf,AutoFakeBold]{simsun.ttc}}
% \setCJKmainfont[Path=fonts/,FallBack=simsunb.ttf,AutoFakeBold]{simsun.ttc} % 中文默认为宋体,小四在\documentclass处设置

% % 黑体
% \renewcommand{\heiti}{\CJKfontspec[Path=fonts/,AutoFakeBold]{simhei.ttf}}
% \setCJKsansfont[Path=fonts/]{simhei.ttf}

% \newcommand{\楷书}{\kaishu}

% \newcommand{\雅黑}{\yahei}


% % 设置章节样式
% \ctexset{
%   % 需要使用ctexart文类，否则报错键不存在
%   section = {
% 		format+ = \zihao{-3} \songti \centering,
% 		name = {,.},
% 		number = \arabic{section},
% 		beforeskip = 1.0ex plus 0.2ex minus .2ex,
% 		% afterskip用于设置章节标题后的垂直间距
% 		afterskip = 1.0ex plus 0.2ex minus .2ex,
% 		aftername = \hspace{0pt}
%   },
%   subsection = {
%     format+ = \zihao{-4} \songti \raggedright,
% 		beforeskip = 1.0ex plus 0.2ex minus .2ex,
% 		afterskip = 1.0ex plus 0.2ex minus .2ex,
% 		aftername = \hspace{0pt}
%   },
%   subsubsection = {
%     format+ = \zihao{-4} \songti \raggedright,
% 		beforeskip = 1.0ex plus 0.2ex minus .2ex,
% 		afterskip = 1.0ex plus 0.2ex minus .2ex,
% 		aftername = \hspace{0pt}
%   },
% }

% % 设置图片文件夹路径
% \graphicspath{{figures/}{figure/}{pictures/}%
% {picture/}{pic/}{pics/}{image/}{images/}}

% % 代码格式和颜色定义
% \usemintedstyle{emacs}

% % 页眉
% \pagestyle{fancy}

% \fancyhead{}


% \setlength{\headheight}{13pt}

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{math}[2023/02/08 v0.2 Standard LaTeX Template for math Report]
\LoadClass[UTF8,a4paper]{ctexart}
\RequirePackage{graphicx}
\RequirePackage{geometry}
\RequirePackage{titlesec}
\RequirePackage{fancyhdr}
\RequirePackage[most,minted]{tcolorbox}
\RequirePackage{minted}
\RequirePackage{amsmath}
\RequirePackage{unicode-math}
\RequirePackage{multirow}
\RequirePackage{array}
\RequirePackage{float}
\RequirePackage{indentfirst}
\RequirePackage{zhnumber}
\RequirePackage{booktabs}
\RequirePackage{todonotes}
\usepackage{fontspec}
\usepackage{setspace}
\usepackage{longtable}
\usepackage{csvsimple}
\usepackage{comment}
\usepackage{graphicx}
\usepackage{array}
\usepackage{makecell}
\usepackage{svg}
\usepackage{subfigure}
\usepackage{array}
\geometry{a4paper,top=2.5cm,bottom=2.5cm,left=2.5cm,right=2.5cm}

\unimathsetup{math-style=ISO, bold-style=ISO}

\RequirePackage{kvdefinekeys}
\RequirePackage{kvsetkeys}
\RequirePackage{kvoptions}

\linespread{1.25}

\xeCJKsetup{EmboldenFactor=2.5}

\setmainfont{Times New Roman}
\renewcommand{\songti}{\CJKfontspec[Path=fonts/,FallBack=simsunb.ttf,AutoFakeBold]{simsun.ttc}}
\setCJKmainfont[Path=fonts/,FallBack=simsunb.ttf,AutoFakeBold]{simsun.ttc}

\renewcommand{\heiti}{\CJKfontspec[Path=fonts/,AutoFakeBold]{simhei.ttf}}
\setCJKsansfont[Path=fonts/]{simhei.ttf}

\newcommand{\楷书}{\kaishu}
\newcommand{\雅黑}{\yahei}

\ctexset{
  section = {
    format+ = \zihao{-3} \songti \centering,
    name = {,.},
    number = \arabic{section},
    beforeskip = 1.0ex plus 0.2ex minus .2ex,
    afterskip = 1.0ex plus 0.2ex minus .2ex,
    aftername = \hspace{0pt}
  },
  subsection = {
    format+ = \zihao{-4} \songti \raggedright,
    beforeskip = 1.0ex plus 0.2ex minus .2ex,
    afterskip = 1.0ex plus 0.2ex minus .2ex,
    aftername = \hspace{0pt}
  },
  subsubsection = {
    format+ = \zihao{-4} \songti \raggedright,
    beforeskip = 1.0ex plus 0.2ex minus .2ex,
    afterskip = 1.0ex plus 0.2ex minus .2ex,
    aftername = \hspace{0pt}
  },
}

\graphicspath{{figures/}{figure/}{pictures/}%
{picture/}{pic/}{pics/}{image/}{images/}}

\usemintedstyle{emacs}

% 页眉
\pagestyle{fancy}
\fancyhead{}
\renewcommand{\headrule}{}

% 添加其他设置和内容
