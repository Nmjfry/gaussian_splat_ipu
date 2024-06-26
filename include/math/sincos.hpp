#pragma once

// Efficient sincos that avoids use of double precision.
// int sincos(float x, float& s, float& c, int flg=0);



#include <cmath>


/* Define one of the following to be 1:
 */
#define ACC5 1
#define ACC11 0
#define ACC17 0

/* Option for linear interpolation when flg = 1
 */
#define LINTERP 1

/* Option for absolute error criterion
 */
#define ABSERR 1

/* Option to include modulo 360 function:
 */
#define MOD360 1

/*
Cephes Math Library Release 2.1
Copyright 1987 by Stephen L. Moshier
Direct inquiries to 30 Frost Street, Cambridge, MA 02140
*/

namespace splat {

/* Table of sin(i degrees)
 * for 0 <= i <= 90
 */
static float sintbl[92] = {
  0.00000000000000000000E0,
  1.74524064372835128194E-2,
  3.48994967025009716460E-2,
  5.23359562429438327221E-2,
  6.97564737441253007760E-2,
  8.71557427476581735581E-2,
  1.04528463267653471400E-1,
  1.21869343405147481113E-1,
  1.39173100960065444112E-1,
  1.56434465040230869010E-1,
  1.73648177666930348852E-1,
  1.90808995376544812405E-1,
  2.07911690817759337102E-1,
  2.24951054343864998051E-1,
  2.41921895599667722560E-1,
  2.58819045102520762349E-1,
  2.75637355816999185650E-1,
  2.92371704722736728097E-1,
  3.09016994374947424102E-1,
  3.25568154457156668714E-1,
  3.42020143325668733044E-1,
  3.58367949545300273484E-1,
  3.74606593415912035415E-1,
  3.90731128489273755062E-1,
  4.06736643075800207754E-1,
  4.22618261740699436187E-1,
  4.38371146789077417453E-1,
  4.53990499739546791560E-1,
  4.69471562785890775959E-1,
  4.84809620246337029075E-1,
  5.00000000000000000000E-1,
  5.15038074910054210082E-1,
  5.29919264233204954047E-1,
  5.44639035015027082224E-1,
  5.59192903470746830160E-1,
  5.73576436351046096108E-1,
  5.87785252292473129169E-1,
  6.01815023152048279918E-1,
  6.15661475325658279669E-1,
  6.29320391049837452706E-1,
  6.42787609686539326323E-1,
  6.56059028990507284782E-1,
  6.69130606358858213826E-1,
  6.81998360062498500442E-1,
  6.94658370458997286656E-1,
  7.07106781186547524401E-1,
  7.19339800338651139356E-1,
  7.31353701619170483288E-1,
  7.43144825477394235015E-1,
  7.54709580222771997943E-1,
  7.66044443118978035202E-1,
  7.77145961456970879980E-1,
  7.88010753606721956694E-1,
  7.98635510047292846284E-1,
  8.09016994374947424102E-1,
  8.19152044288991789684E-1,
  8.29037572555041692006E-1,
  8.38670567945424029638E-1,
  8.48048096156425970386E-1,
  8.57167300702112287465E-1,
  8.66025403784438646764E-1,
  8.74619707139395800285E-1,
  8.82947592858926942032E-1,
  8.91006524188367862360E-1,
  8.98794046299166992782E-1,
  9.06307787036649963243E-1,
  9.13545457642600895502E-1,
  9.20504853452440327397E-1,
  9.27183854566787400806E-1,
  9.33580426497201748990E-1,
  9.39692620785908384054E-1,
  9.45518575599316810348E-1,
  9.51056516295153572116E-1,
  9.56304755963035481339E-1,
  9.61261695938318861916E-1,
  9.65925826289068286750E-1,
  9.70295726275996472306E-1,
  9.74370064785235228540E-1,
  9.78147600733805637929E-1,
  9.81627183447663953497E-1,
  9.84807753012208059367E-1,
  9.87688340595137726190E-1,
  9.90268068741570315084E-1,
  9.92546151641322034980E-1,
  9.94521895368273336923E-1,
  9.96194698091745532295E-1,
  9.97564050259824247613E-1,
  9.98629534754573873784E-1,
  9.99390827019095730006E-1,
  9.99847695156391239157E-1,
  1.00000000000000000000E0,
  9.99847695156391239157E-1,
};

static int sincos(float x, float& s, float& c, int flg=0) {
  int ix, ssign, csign, xsign;
  float y, z, sx, sz, cx, cz;

  x = x * float(180.0 / 3.14159265358979323846264338327950288);
  /* Make argument nonnegative.
  */
  xsign = 1;
  if( x < 0.f )
    {
    xsign = -1;
    x = -x;
    }


  #if MOD360
  x = x  -  360.f * std::floor( x / 360.f );
  #endif

  /* Find nearest integer to x.
  * Note there should be a domain error test here,
  * but this is omitted to gain speed.
  */
  ix = x + .5f;
  z = x - ix;        /* the residual */

  /* Look up the sine and cosine of the integer.
  */
  if( ix <= 180 )
    {
    ssign = 1;
    csign = 1;
    }
  else
    {
    ssign = -1;
    csign = -1;
    ix -= 180;
    }

  if( ix > 90 )
    {
    csign = -csign;
    ix = 180 - ix;
    }

  sx = sintbl[ix];
  if( ssign < 0 )
    sx = -sx;
  cx = sintbl[ 90-ix ];
  if( csign < 0 )
    cx = -cx;

  /* If the flag argument is set, then just return
  * the tabulated values for arg to the nearest whole degree.
  */
  if( flg )
    {
  #if LINTERP
    y = sx + 1.74531263774940077459e-2f * z * cx;
    cx -= 1.74531263774940077459e-2f * z * sx;
    sx = y;
  #endif
    if( xsign < 0 )
      sx = -sx;
    s = sx;    /* sine */
    c = cx;    /* cosine */
    return 0;
    }

  /* Find sine and cosine
  * of the residual angle between -0.5 and +0.5 degree.
  */
  #if ACC5
  #if ABSERR
  /* absolute error = 2.769e-8: */
  sz = 1.74531263774940077459e-2f * z;
  /* absolute error = 4.146e-11: */
  cz = 1.f - 1.52307909153324666207e-4f * z * z;
  #else
  /* relative error = 6.346e-6: */
  sz = 1.74531817576426662296e-2f * z;
  /* relative error = 3.173e-6: */
  cz = 1.f - 1.52308226602566149927e-4f * z * z;
  #endif
  #else
  y = z * z;
  #endif


  #if ACC11
  sz = ( -8.86092781698004819918e-7f * y
        + 1.74532925198378577601e-2f     ) * z;

  cz = 1.f - ( -3.86631403698859047896e-9f * y
              + 1.52308709893047593702e-4f     ) * y;
  #endif


  #if ACC17
  sz = ((  1.34959795251974073996e-11f * y
        - 8.86096155697856783296e-7f     ) * y
        + 1.74532925199432957214e-2f         ) * z;

  cz = 1.f - ((  3.92582397764340914444e-14f * y
              - 3.86632385155548605680e-9f     ) * y
              + 1.52308709893354299569e-4f          ) * y;
  #endif

  /* Combine the tabulated part and the calculated part
  * by trigonometry.
  */
  y = sx * cz  +  cx * sz;
  if( xsign < 0 )
    y = - y;
  s = y; /* sine */

  c = cx * cz  -  sx * sz; /* cosine */
  return 0;
}

} // end of namespace splat