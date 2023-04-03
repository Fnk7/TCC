%0 = tc.add(%A, %B) {axesmaps = [(0),(0)]}
%1 = tc.add(%C, %D) {axesmaps = [(0),(0)]}
%r = tc.mul(%0, %1) {axesmaps = [(0),(0)]}
=>
%1 = tc.add(%C, %D) {axesmaps = [(0),(0)]}
%2 = tc.mul(%A, %1) {axesmaps = [(0),(0)]}
%3 = tc.mul(%B, %1) {axesmaps = [(0),(0)]}
%r = tc.add(%2, %3) {axesmaps = [(0),(0)]}
=>
%0 = tc.mul(%A, %C) {axesmaps = [(0),(0)]}
%1 = tc.mul(%A, %D) {axesmaps = [(0),(0)]}
%2 = tc.mul(%B, %C) {axesmaps = [(0),(0)]}
%3 = tc.mul(%B, %D) {axesmaps = [(0),(0)]}
%r = tc.add(%0, %1, %2, %3) {axesmaps = [(0),(0),(0),(0)]}

