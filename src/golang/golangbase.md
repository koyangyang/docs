---
title: Golang语言记录
weight: 2
---
## Go语言基础

### 数组

```go
var array = []int{1,2,3}
```

### 切片(slice,列表)

```go
var array []string array = append(array, "a") array = append(array, "b") array = append(array, "c")
```

### make函数

除了基本数据类型，其他数据类型如果只定义不赋值，那么实际的值就是`nil`

```go
var list []string fmt.Println(list == nil) // true var list = make([]string, 0) fmt.Println(list == nil) // false
```

### map

map的key必须是基本数据类型，value可以是任意类型,`无序`

```go
// 声明 var m1 = map[string]string{ "age": 21, } var m2 = make(map[string]string)
```

### for循环

```go
for 初始化;条件;操作{ } var sum = 0 for i := 0; i <= 100; i++ { sum += i } // 遍历列表 for index,i := range slice{ fmt.Println(index,i) } // 遍历map for key,value := range map{ fmt.Println(key,value) }
```

### 函数参数

```go
func Add(numlist ...int) int { sum := 0 for _, num := range numlist { sum += num } return sum } func main(){ sum := Add(1, 2, 3, 4, 5) fmt.Println(sum) }
```

### 匿名函数

```go
package main import "fmt" func main() { var add = func(a, b int) int { return a + b } fmt.Println(add(1, 2)) }
```

### 高阶函数

```go
func main() { var num int fmt.Scan(&num) var funcMap = map[int]func(){ 1: func() { fmt.Println("登录") }, 2: func() { fmt.Println("个人中心") }, 3: func() { fmt.Println("注销") }, } funcMap[num]() }
```

### 闭包

设计一个函数，先传一个参数表示延时，后面再次传参数就是将参数求和

```go
package main import ( "fmt" "time" ) func awaitAdd(t int) func(...int) int { time.Sleep(time.Duration(t) * time.Second) return func(numList ...int) int { var sum int for _, i2 := range numList { sum += i2 } return sum } } func main() { fmt.Println(awaitAdd(2)(1, 2, 3)) }
```

### 传址调用

```go
func change(num *int){ *num = 2 } func main(){ num := 3 change(&num) }
```

### 结构体

```go
// Student 定义结构体 type Student struct { Name string Age int } // PrintInfo 给机构体绑定一个方法 func (s Student) PrintInfo() { fmt.Printf("name:%s age:%d\n", s.Name, s.Age) }
```

### 继承

```go
type People struct { Time string } func (p People) Info() { fmt.Println("people ", p.Time) } // Student 定义结构体 type Student struct { People Name string Age int } // PrintInfo 给机构体绑定一个方法 func (s Student) PrintInfo() { fmt.Printf("name:%s age:%d\n", s.Name, s.Age) } func main() { p := People{ Time: "2023-11-15 14:51", } s := Student{ People: p, Name: "枫枫", Age: 21, } s.Name = "枫枫知道" // 修改值 s.PrintInfo() s.Info() // 可以调用父结构体的方法 fmt.Println(s.People.Time) // 调用父结构体的属性 fmt.Println(s.Time) // 也可以这样 }
```

### 结构体tag

```go
package main import ( "encoding/json" "fmt" ) type Student struct { Name string `json:"name"` Age int `json:"age,omitempty"`// omitempty,如果这个字段是空值不会被json转出来 Password string `json:"-"` //这个字段不会被json转出来 } func main() { s := Student{ Name: "枫枫", Age: 21, Password: "123456", } byteData, _ := json.Marshal(s) fmt.Println(string(byteData)) // {"name":"枫枫","age":21} }
```

### 自定义数据类型

```go
type Code int const ( SuccessCode Code = 0 ValidCode Code = 7 // 校验失败的错误 ServiceErrCode Code = 8 // 服务错误 )
```

### 接口

```go
package main import "fmt" // Animal 定义一个animal的接口，它有唱，跳，rap的方法 type Animal interface { sing() jump() rap() } // Chicken 需要全部实现这些接口 type Chicken struct { Name string } func (c Chicken) sing() { fmt.Println("chicken 唱") } func (c Chicken) jump() { fmt.Println("chicken 跳") } func (c Chicken) rap() { fmt.Println("chicken rap") } // 全部实现完之后，chicken就不再是一只普通的鸡了 func main() { var animal Animal animal = Chicken{"ik"} animal.sing() animal.jump() animal.rap() }
```

### 协程

```go
package main import ( "fmt" "sync" "time" ) var moneyChan = make(chan int) // 声明并初始化一个长度为0的信道 func pay(name string, money int, wg *sync.WaitGroup) { defer wg.Done() fmt.Printf("%s 开始购物 %d\n", name, money) time.Sleep(2 * time.Second) fmt.Printf("%s 购物结束\n", name) moneyChan <- money } // 协程 func main() { var wait sync.WaitGroup wait.Add(3) // 主线程结束，协程函数跟着结束 go pay("张三", 200, &wait) go pay("王五", 300, &wait) go pay("李四", 500, &wait) go func() { defer close(moneyChan) // 在协程函数里面等待上面三个协程函数结束 wait.Wait() }() for money := range moneyChan { fmt.Println("收到", money) } }
```

#### 协程取多个channel

```go
package main import ( "fmt" "sync" "time" ) var moneyChan = make(chan int) // 声明并初始化一个长度为0的信道 var nameChan = make(chan string) var doneChan = make(chan struct{}) func pay(name string, money int, wg *sync.WaitGroup) { defer wg.Done() fmt.Printf("%s 开始购物 %d\n", name, money) time.Sleep(2 * time.Second) fmt.Printf("%s 购物结束\n", name) moneyChan <- money nameChan <- name } // 协程 func main() { var wait sync.WaitGroup wait.Add(3) // 主线程结束，协程函数跟着结束 go pay("张三", 200, &wait) go pay("王五", 300, &wait) go pay("李四", 500, &wait) go func() { defer close(moneyChan) defer close(nameChan) defer close(doneChan) // 在协程函数里面等待上面三个协程函数结束 wait.Wait() }() var event = func() { for { select { case money := <-moneyChan: fmt.Printf("money: %d\n", money) case name := <-nameChan: fmt.Printf("name: %s\n", name) case <-doneChan: return } } } event() }
```

#### 协程超时处理

```go
package main import ( "fmt" "time" ) var doneChan = make(chan struct{}) func event() { fmt.Println("event start") time.Sleep(2 * time.Second) fmt.Println("event end") close(doneChan) } func main() { go event() select { case <-doneChan: fmt.Println("done") case <-time.After(1 * time.Second): fmt.Println("timeout") return } }
```

### 同步锁

```go
package main import ( "fmt" "sync" ) var num int var wait sync.WaitGroup var lock sync.Mutex func add() { // 谁先抢到了这把锁，谁就把它锁上，一旦锁上，其他的线程就只能等着 lock.Lock() for i := 0; i < 1000000; i++ { num++ } lock.Unlock() wait.Done() } func reduce() { lock.Lock() for i := 0; i < 1000000; i++ { num-- } lock.Unlock() wait.Done() } func main() { wait.Add(2) go add() go reduce() wait.Wait() fmt.Println(num) }
```

#### 同步锁map

```go
package main import ( "fmt" "sync" "time" ) var wait sync.WaitGroup var mp = sync.Map{} func reader() { for { fmt.Println(mp.Load("time")) } wait.Done() } func writer() { for { mp.Store("time", time.Now().Format("15:04:05")) } wait.Done() } func main() { wait.Add(2) go reader() go writer() wait.Wait() }
```

## Go网络请求

### Get请求

```go
package utils import ( "fmt" "net/http" ) func PushplusGet() { url := "http://www.xxx.com" client, _ := http.Get(url) fmt.Println(client.StatusCode) }
```

### post请求

#### application/x-www-form-urlencoded

```go
package utils import ( "bytes" "encoding/json" "fmt" "net/http" "testing" ) func TestPushplusPost(t *testing.T) { param := url.Values{} param.Add("token", "20efb508d9ef4ec484115b2d7682e554") param.Add("title", "Goland_Test_Post") param.Add("content", "test") param.Add("template", "markdown") pushplus_url := "http://www.pushplus.plus/send" res, _ :=http.PostForm(pushplus_url, param) byteData, _ = io.ReadAll(res.Body) //返回json格式数据 var data map[string]any json.Unmarshal(byteData, &data) fmt.Println(data["code"]) }
```

#### multipart/form-data

```go
package utils import ( "bytes" "encoding/json" "fmt" "net/http" "testing" ) func TestPushplusPost(t *testing.T) { param := url.Values{} param.Add("token", "20efb508d9ef4ec484115b2d7682e554") param.Add("title", "Goland_Test_Post") param.Add("content", "test") param.Add("template", "markdown") pushplus_url := "http://www.pushplus.plus/send" res, _ :=http.PostForm(pushplus_url, param) byteData, _ = io.ReadAll(res.Body) //返回json格式数据 var data map[string]any json.Unmarshal(byteData, &data) fmt.Println(data["code"]) }
```

#### application/json

```go
package utils import ( "bytes" "encoding/json" "fmt" "net/http" "testing" ) func TestPushplusPost(t *testing.T) { byteData, _ := json.Marshal(map[string]string{ "token": "20efb508d9ef4ec484115b2d7682e554", "title": "Goland_Test_Post", "content": "test", "template": "markdown", }) pushplus_url := "http://www.pushplus.plus/send" res, _ := http.Post(pushplus_url, "application/json", bytes.NewReader(byteData)) byteData, _ = io.ReadAll(res.Body) //返回json格式数据 var data map[string]any json.Unmarshal(byteData, &data) fmt.Println(data["code"]) }
```

### 单元测试