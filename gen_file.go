package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
)

func main() {
	fileName := "train.txt"
	file, err := os.OpenFile(fileName, os.O_RDWR|os.O_CREATE, 0755)
	if err != nil {
		log.Fatal(err)
	}
	input := bufio.NewWriter(file)
	for i := 0; i <= 102; i++ {
		input.WriteString(fmt.Sprintf("r%04d.png s%04d.png\n", i, i))
	}
	input.Flush()

	//input.WriteString()
}
