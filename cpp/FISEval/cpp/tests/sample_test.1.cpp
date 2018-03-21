// 010-TestCase.cpp

#include "../vendor/catch/catch.hpp"

int Factorial3( int number ) {
   return number <= 1 ? number : Factorial3( number - 1 ) * number;  // fail
// return number <= 1 ? 1      : Factorial3( number - 1 ) * number;  // pass
}

TEST_CASE( "Factorial3 of 0 is 1 (fail)", "[single-file]" ) {
    REQUIRE( Factorial3(0) == 1 );
}

TEST_CASE( "Factorial3s of 1 and higher are computed (pass)", "[single-file]" ) {
    REQUIRE( Factorial3(1) == 1 );
    REQUIRE( Factorial3(2) == 2 );
    REQUIRE( Factorial3(3) == 6 );
    REQUIRE( Factorial3(10) == 3628800 );
}

// Compile & run:
// - g++ -std=c++11 -Wall -I$(CATCH_SINGLE_INCLUDE) -o 010-TestCase 010-TestCase.cpp && 010-TestCase --success
// - cl -EHsc -I%CATCH_SINGLE_INCLUDE% 010-TestCase.cpp && 010-TestCase --success

// Expected compact output (all assertions):
//
// prompt> 010-TestCase --reporter compact --success
// 010-TestCase.cpp:14: failed: Factorial3(0) == 1 for: 0 == 1
// 010-TestCase.cpp:18: passed: Factorial3(1) == 1 for: 1 == 1
// 010-TestCase.cpp:19: passed: Factorial3(2) == 2 for: 2 == 2
// 010-TestCase.cpp:20: passed: Factorial3(3) == 6 for: 6 == 6
// 010-TestCase.cpp:21: passed: Factorial3(10) == 3628800 for: 3628800 (0x375f00) == 3628800 (0x375f00)
// Failed 1 test case, failed 1 assertion.
